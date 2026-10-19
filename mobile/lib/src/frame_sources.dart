import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:sign_landmarks/sign_landmarks.dart';

/// One camera frame, ready for SignLandmarks.detect().
class SourceFrame {
  SourceFrame({
    required this.bytes,
    required this.format,
    this.width = 0,
    this.height = 0,
    this.bytesPerRow = 0,
    this.rotation = 0,
  });

  final Uint8List bytes;
  final FrameFormat format;
  final int width;
  final int height;
  final int bytesPerRow;

  /// Clockwise rotation that makes the frame upright.
  final int rotation;
}

abstract class FrameSource {
  /// Frames are pushed as they arrive; slow consumers should drop frames.
  Stream<SourceFrame> get frames;

  /// Whether the on-screen preview of this source is already mirrored (selfie view).
  bool get previewMirrored;

  /// Rotation that makes this source's frames upright. Used for drawing; already applied to frames.
  int get rotation;

  Future<void> start();
  Future<void> stop();
}

/// Phone camera via the camera plugin's image stream (NV21 on Android, BGRA on iOS).
class PhoneCameraSource implements FrameSource {
  PhoneCameraSource({this.front = false, this.rotationOverride = -1});

  final bool front;
  final int rotationOverride;
  final _frames = StreamController<SourceFrame>.broadcast();
  CameraController? controller;

  @override
  Stream<SourceFrame> get frames => _frames.stream;

  // CameraPreview mirrors the front camera only.
  @override
  bool get previewMirrored => front;

  @override
  int get rotation => rotationOverride >= 0 ? rotationOverride : (controller?.description.sensorOrientation ?? 0);

  @override
  Future<void> start() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) throw StateError('No camera available');
    final wanted = front ? CameraLensDirection.front : CameraLensDirection.back;
    final description = cameras.firstWhere((c) => c.lensDirection == wanted, orElse: () => cameras.first);
    final c = CameraController(
      description,
      ResolutionPreset.medium, // 640x480-class; frames are downscaled further to the profile width natively
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
    );
    await c.initialize();
    controller = c;
    await c.startImageStream((image) {
      if (!_frames.hasListener) return;
      _frames.add(Platform.isAndroid
          ? SourceFrame(
              bytes: toNv21(image),
              format: FrameFormat.nv21,
              width: image.width,
              height: image.height,
              rotation: rotation,
            )
          : SourceFrame(
              bytes: image.planes[0].bytes,
              format: FrameFormat.bgra8888,
              width: image.width,
              height: image.height,
              bytesPerRow: image.planes[0].bytesPerRow,
              rotation: rotation,
            ));
    });
  }

  @override
  Future<void> stop() async {
    final c = controller;
    controller = null;
    if (c == null) return;
    if (c.value.isStreamingImages) await c.stopImageStream();
    await c.dispose();
  }
}

/// Packs an Android camera image into tightly packed NV21 (Y plane + interleaved VU).
/// Handles both a single NV21 plane and three YUV_420_888 planes with row/pixel strides.
Uint8List toNv21(CameraImage image) {
  final width = image.width;
  final height = image.height;
  final out = Uint8List(width * height * 3 ~/ 2);
  final yPlane = image.planes[0];
  for (var row = 0; row < height; row++) {
    out.setRange(row * width, row * width + width, yPlane.bytes, row * yPlane.bytesPerRow);
  }
  var o = width * height;
  if (image.planes.length == 1) {
    // Single NV21 buffer: the VU block follows the Y block at the same row stride.
    final stride = yPlane.bytesPerRow;
    final vuStart = stride * height;
    for (var row = 0; row < height ~/ 2; row++) {
      out.setRange(o, o + width, yPlane.bytes, vuStart + row * stride);
      o += width;
    }
    return out;
  }
  final u = image.planes[1];
  final v = image.planes[2];
  final pixelStride = v.bytesPerPixel ?? 1;
  for (var row = 0; row < height ~/ 2; row++) {
    for (var col = 0; col < width ~/ 2; col++) {
      final vi = row * v.bytesPerRow + col * pixelStride;
      final ui = row * u.bytesPerRow + col * (u.bytesPerPixel ?? 1);
      out[o++] = v.bytes[vi];
      out[o++] = u.bytes[ui];
    }
  }
  return out;
}

/// The glasses camera: an MJPEG HTTP stream (multipart/x-mixed-replace), e.g. from an ESP32-S3.
class MjpegSource implements FrameSource {
  MjpegSource(this.url, {this.rotationOverride = -1});

  final String url;
  final int rotationOverride;
  final _frames = StreamController<SourceFrame>.broadcast();
  final _parser = MjpegParser();
  http.Client? _client;
  StreamSubscription<List<int>>? _sub;
  bool _running = false;

  /// Latest JPEG, for the on-screen preview.
  Uint8List? lastJpeg;

  @override
  Stream<SourceFrame> get frames => _frames.stream;

  @override
  bool get previewMirrored => false;

  @override
  int get rotation => rotationOverride >= 0 ? rotationOverride : 0;

  @override
  Future<void> start() async {
    _running = true;
    await _connect();
  }

  Future<void> _connect() async {
    _client?.close();
    final client = http.Client();
    _client = client;
    final res = await client.send(http.Request('GET', Uri.parse(url))).timeout(const Duration(seconds: 6));
    if (res.statusCode != 200) throw HttpException('Glasses stream returned ${res.statusCode}');
    _sub = res.stream.listen(
      (chunk) {
        for (final jpeg in _parser.add(chunk)) {
          lastJpeg = jpeg;
          if (_frames.hasListener) {
            _frames.add(SourceFrame(bytes: jpeg, format: FrameFormat.jpeg, rotation: rotation));
          }
        }
      },
      onError: (_) => _reconnect(),
      onDone: _reconnect,
      cancelOnError: true,
    );
  }

  Future<void> _reconnect() async {
    if (!_running) return;
    await Future<void>.delayed(const Duration(seconds: 1));
    if (!_running) return;
    try {
      await _connect();
    } catch (_) {
      unawaited(_reconnect());
    }
  }

  @override
  Future<void> stop() async {
    _running = false;
    await _sub?.cancel();
    _client?.close();
    _client = null;
  }
}

/// Extracts complete JPEG images (SOI 0xFFD8 ... EOI 0xFFD9) from an MJPEG byte stream,
/// independent of the multipart boundary format.
class MjpegParser {
  final _buffer = BytesBuilder(copy: false);
  static const maxBuffer = 4 * 1024 * 1024;

  List<Uint8List> add(List<int> chunk) {
    _buffer.add(chunk);
    var data = _buffer.toBytes();
    final frames = <Uint8List>[];
    var start = _indexOf(data, 0xD8, 0);
    while (start >= 0) {
      final end = _indexOf(data, 0xD9, start + 2);
      if (end < 0) break;
      frames.add(Uint8List.sublistView(data, start, end + 2));
      start = _indexOf(data, 0xD8, end + 2);
    }
    _buffer.clear();
    if (start >= 0) {
      _buffer.add(Uint8List.sublistView(data, start));
    } else if (data.isNotEmpty && data.last == 0xFF) {
      _buffer.add([0xFF]); // a marker may be split across chunks
    }
    if (_buffer.length > maxBuffer) _buffer.clear(); // corrupt stream: resync
    return frames;
  }

  /// Index of the 0xFF byte of the next 0xFF [marker] pair at or after [from], or -1.
  static int _indexOf(Uint8List data, int marker, int from) {
    for (var i = from; i < data.length - 1; i++) {
      if (data[i] == 0xFF && data[i + 1] == marker) return i;
    }
    return -1;
  }
}
