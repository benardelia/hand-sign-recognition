import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:sign_landmarks/sign_landmarks.dart';

import 'frame_sources.dart';
import 'gloss_decoder.dart';
import 'model_store.dart';
import 'sign_classifier.dart';

enum RecordState { idle, countdown, recording, uploading, saved, error }

/// camera frames -> MediaPipe landmarks (native) -> sliding window -> TFLite -> sentence.
/// Frames are paced to the profile fps and dropped while the previous one is processing.
class TranslationPipeline extends ChangeNotifier {
  TranslationPipeline({required this.classifier, required double threshold, this.onSentence})
      : decoder = GlossDecoder(classifier.model.labels, threshold: threshold);

  final SignClassifier classifier;
  final GlossDecoder decoder;
  final void Function(String sentence)? onSentence;

  FrameSource? _source;
  StreamSubscription<SourceFrame>? _sub;
  final List<Float32List> _window = [];
  bool _busy = false;
  int _lastRun = 0;
  int _lastFrameAt = 0;
  final Stopwatch _clock = Stopwatch()..start();

  double fps = 0;
  Float32List? lastFrame;
  String? currentWord;
  String error = '';

  // Recording (collect mode)
  RecordState recordState = RecordState.idle;
  String recordLabel = '';
  String recordMessage = '';
  int countdown = 0;
  final List<Float32List> _recFrames = [];
  final List<int> _recTimes = [];
  int _recStart = 0;

  ProfileModel get model => classifier.model;

  Future<void> start(FrameSource source) async {
    await stop();
    _source = source;
    _sub = source.frames.listen(_onFrame);
    await source.start();
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
    await _source?.stop();
    _source = null;
    _window.clear();
  }

  void clear() {
    decoder.reset('Sentence cleared.');
    _window.clear();
    onSentence?.call('');
    notifyListeners();
  }

  Future<void> _onFrame(SourceFrame frame) async {
    final now = _clock.elapsedMilliseconds;
    if (_busy || now - _lastRun < 1000 ~/ model.cameraFps - 2) return;
    _busy = true;
    _lastRun = now;
    try {
      final values = await SignLandmarks.detect(
        bytes: frame.bytes,
        format: frame.format,
        width: frame.width,
        height: frame.height,
        bytesPerRow: frame.bytesPerRow,
        rotationDegrees: frame.rotation,
        mirror: true,
        targetWidth: model.cameraWidth,
        timestampMs: now,
      );
      if (_lastFrameAt > 0) fps = 0.9 * fps + 0.1 * (1000 / (now - _lastFrameAt));
      _lastFrameAt = now;
      lastFrame = values;
      error = '';
      await _handleFrame(values, now);
    } catch (e) {
      error = '$e';
    } finally {
      _busy = false;
      notifyListeners();
    }
  }

  Future<void> _handleFrame(Float32List values, int now) async {
    if (recordState == RecordState.recording) {
      _recFrames.add(values);
      _recTimes.add(now);
      final elapsed = now - _recStart;
      if ((elapsed >= _recordMs && _recFrames.length >= _minRecordFrames) || elapsed >= 3 * _recordMs) {
        unawaited(_finishRecording());
      }
      return;
    }
    if (recordState != RecordState.idle) return;

    _window.add(values);
    if (_window.length > model.sequenceLength) _window.removeAt(0);
    if (_window.length < model.sequenceLength) return;
    final before = decoder.sentence;
    currentWord = decoder.update(await classifier.predict(_window));
    if (decoder.sentence != before) onSentence?.call(decoder.sentence);
  }

  // --- Collect mode: record a sample and upload it to the hub ---
  static const _recordMs = 1000; // config/profiles.json dataset.record_seconds
  static const _minRecordFrames = 8;
  String _serverUrl = '';
  String _token = '';
  String _sourceName = 'phone';

  Future<void> record(String label, {required String serverUrl, required String token, required String source}) async {
    if (recordState != RecordState.idle) return;
    _serverUrl = serverUrl;
    _token = token;
    _sourceName = source;
    recordLabel = label.trim().toUpperCase();
    recordState = RecordState.countdown;
    for (countdown = 3; countdown > 0; countdown--) {
      notifyListeners();
      await Future<void>.delayed(const Duration(seconds: 1));
    }
    _recFrames.clear();
    _recTimes.clear();
    _recStart = _clock.elapsedMilliseconds;
    recordState = RecordState.recording;
    notifyListeners();
  }

  Future<void> _finishRecording() async {
    if (_recFrames.length < _minRecordFrames) {
      _endRecording(RecordState.error, 'Only ${_recFrames.length} frames: device too slow. Try the LOW profile.');
      return;
    }
    recordState = RecordState.uploading;
    notifyListeners();
    final buffer = Float32List(_recFrames.length * SignClassifier.frameSize);
    for (var i = 0; i < _recFrames.length; i++) {
      buffer.setRange(i * SignClassifier.frameSize, (i + 1) * SignClassifier.frameSize, _recFrames[i]);
    }
    final t0 = _recTimes.first;
    final seconds = (_recTimes.last - t0) / 1000;
    final body = jsonEncode({
      'label': recordLabel,
      'frames_b64': base64Encode(buffer.buffer.asUint8List()),
      'meta': {
        'source': _sourceName,
        'profile': model.name,
        'mirrored': true,
        'fps': seconds > 0 ? ((_recFrames.length - 1) / seconds * 10).round() / 10 : null,
        'timestamps_ms': [for (final t in _recTimes) t - t0],
        'resolution_width': model.cameraWidth,
        'app': 'scribesign_glasses',
      },
    });
    try {
      final res = await http.post(
        Uri.parse('$_serverUrl/api/samples'),
        headers: {'Content-Type': 'application/json', if (_token.isNotEmpty) 'X-Handsign-Token': _token},
        body: body,
      );
      final json = jsonDecode(res.body) as Map<String, dynamic>;
      if (res.statusCode == 200) {
        _endRecording(RecordState.saved, 'Saved ${_recFrames.length} frames for $recordLabel');
      } else {
        _endRecording(RecordState.error, json['message'] as String? ?? 'Upload failed (${res.statusCode})');
      }
    } catch (e) {
      _endRecording(RecordState.error, 'Upload failed: $e');
    }
  }

  void _endRecording(RecordState state, String message) {
    recordState = state;
    recordMessage = message;
    _recFrames.clear();
    notifyListeners();
    Future<void>.delayed(const Duration(seconds: 2), () {
      recordState = RecordState.idle;
      notifyListeners();
    });
  }

  bool _disposed = false;

  // A frame that was in flight when the pipeline stopped may finish after dispose().
  @override
  void notifyListeners() {
    if (!_disposed) super.notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    super.dispose();
  }

  /// Stops the camera and releases the native classifier. Await it before loading another
  /// model: the plugin holds a single classifier.
  Future<void> shutdown() async {
    await stop();
    await classifier.close();
  }
}
