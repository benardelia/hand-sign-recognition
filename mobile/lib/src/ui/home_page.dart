import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:sign_landmarks/sign_landmarks.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../frame_sources.dart';
import '../glasses_link.dart';
import '../model_store.dart';
import '../settings.dart';
import '../sign_classifier.dart';
import '../translation_pipeline.dart';
import 'landmark_painter.dart';
import 'settings_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.settings});

  final AppSettings settings;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _store = ModelStore();
  final _glasses = GlassesLink();
  TranslationPipeline? _pipeline;
  FrameSource? _source;
  String? _loading = 'Starting...';
  double? _progress;
  String? _error;
  String _delegate = '';

  AppSettings get s => widget.settings;

  @override
  void initState() {
    super.initState();
    unawaited(_setup());
  }

  @override
  void dispose() {
    unawaited(WakelockPlus.disable());
    final pipeline = _pipeline;
    if (pipeline != null) unawaited(pipeline.shutdown().then((_) => pipeline.dispose()));
    unawaited(_glasses.disconnect());
    super.dispose();
  }

  void _onProgress(String message, double? progress) {
    if (!mounted) return;
    setState(() {
      _loading = message;
      _progress = progress;
    });
  }

  Future<void> _setup() async {
    final old = _pipeline;
    setState(() {
      _pipeline = null;
      _source = null;
      _error = null;
      _loading = 'Starting...';
    });
    if (old != null) {
      await old.shutdown();
      old.dispose();
    }
    try {
      final holistic = await _store.ensureHolisticModel(onProgress: _onProgress);
      _onProgress('Loading tracking model', null);
      _delegate = await SignLandmarks.initialize(modelPath: holistic, useGpu: s.useGpu);
      final model = await _store.ensureProfile(s.serverUrl, s.profile, onProgress: _onProgress);
      final pipeline = TranslationPipeline(
        classifier: await SignClassifier.load(model),
        threshold: s.threshold,
        onSentence: (text) => unawaited(_glasses.send(text)),
      );
      final source = s.source == InputSource.phone
          ? PhoneCameraSource(front: s.useFrontCamera, rotationOverride: s.rotationOverride)
          : MjpegSource(_glasses.streamUrl ?? s.glassesStreamUrl, rotationOverride: s.rotationOverride);
      _onProgress('Starting camera', null);
      await pipeline.start(source);
      await WakelockPlus.enable();
      if (!mounted) {
        await pipeline.shutdown();
        pipeline.dispose();
        return;
      }
      setState(() {
        _pipeline = pipeline;
        _source = source;
        _loading = null;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = null;
          _error = '$e';
        });
      }
    }
  }

  Future<void> _openSettings() async {
    await Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => SettingsPage(settings: s)));
    await _setup();
  }

  Future<void> _connectGlasses() async {
    if (_glasses.connected) {
      await _glasses.disconnect();
      return;
    }
    try {
      await _glasses.connect();
      final url = _glasses.streamUrl;
      if (url != null && url != s.glassesStreamUrl) {
        s.glassesStreamUrl = url;
        if (s.source == InputSource.glasses) await _setup();
      }
      final sentence = _pipeline?.decoder.sentence;
      if (sentence != null) await _glasses.send(sentence);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Glasses: $e')));
      }
    }
  }

  Future<void> _record() async {
    final pipeline = _pipeline;
    if (pipeline == null) return;
    final controller = TextEditingController(text: pipeline.recordLabel);
    final label = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Record a training sample'),
        content: TextField(
          controller: controller,
          autofocus: true,
          textCapitalization: TextCapitalization.characters,
          decoration: const InputDecoration(labelText: 'Sign label (e.g. THANK YOU)'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, controller.text), child: const Text('Record')),
        ],
      ),
    );
    if (label == null || label.trim().isEmpty) return;
    await pipeline.record(label,
        serverUrl: s.serverUrl, token: s.collectToken, source: s.source == InputSource.glasses ? 'glasses' : 'phone');
  }

  @override
  Widget build(BuildContext context) {
    final pipeline = _pipeline;
    return Scaffold(
      appBar: AppBar(
        title: const Text('ScribeSign'),
        actions: [
          ListenableBuilder(
            listenable: _glasses,
            builder: (context, _) => IconButton(
              tooltip: _glasses.status,
              icon: Icon(_glasses.connected ? Icons.bluetooth_connected : Icons.bluetooth),
              onPressed: _connectGlasses,
            ),
          ),
          IconButton(icon: const Icon(Icons.settings), onPressed: _openSettings),
        ],
      ),
      body: SafeArea(
        child: pipeline == null
            ? _StatusView(loading: _loading, progress: _progress, error: _error, onRetry: _setup)
            : ListenableBuilder(
                listenable: pipeline,
                builder: (context, _) => _TranslatorView(
                  pipeline: pipeline,
                  source: _source!,
                  delegate: _delegate,
                  glasses: _glasses,
                  onClear: pipeline.clear,
                  onRecord: _record,
                ),
              ),
      ),
    );
  }
}

class _StatusView extends StatelessWidget {
  const _StatusView({required this.loading, required this.progress, required this.error, required this.onRetry});

  final String? loading;
  final double? progress;
  final String? error;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: error != null
              ? [
                  const Icon(Icons.error_outline, size: 48, color: Colors.redAccent),
                  const SizedBox(height: 12),
                  Text(error!, textAlign: TextAlign.center),
                  const SizedBox(height: 16),
                  FilledButton(onPressed: onRetry, child: const Text('Retry')),
                ]
              : [
                  SizedBox(width: 240, child: LinearProgressIndicator(value: progress)),
                  const SizedBox(height: 12),
                  Text(loading ?? ''),
                ],
        ),
      ),
    );
  }
}

class _TranslatorView extends StatelessWidget {
  const _TranslatorView({
    required this.pipeline,
    required this.source,
    required this.delegate,
    required this.glasses,
    required this.onClear,
    required this.onRecord,
  });

  final TranslationPipeline pipeline;
  final FrameSource source;
  final String delegate;
  final GlassesLink glasses;
  final VoidCallback onClear;
  final VoidCallback onRecord;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final word = pipeline.currentWord;
    return Column(
      children: [
        Expanded(child: Center(child: _preview())),
        if (pipeline.error.isNotEmpty)
          Padding(
            padding: const EdgeInsets.all(8),
            child: Text(pipeline.error, style: const TextStyle(color: Colors.redAccent), maxLines: 2),
          ),
        Card(
          margin: const EdgeInsets.all(12),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(pipeline.decoder.sentence, style: theme.textTheme.headlineSmall),
                const SizedBox(height: 8),
                Text(
                  word == null
                      ? 'Watching... (${(pipeline.decoder.confidence * 100).round()}%)'
                      : '$word  ${(pipeline.decoder.confidence * 100).round()}%',
                  style: theme.textTheme.bodyMedium,
                ),
              ],
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Text(
            '${pipeline.model.name.toUpperCase()} v${pipeline.model.version} · $delegate · '
            '${pipeline.fps.toStringAsFixed(0)} fps · ${glasses.status}',
            style: theme.textTheme.bodySmall,
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(onPressed: onClear, icon: const Icon(Icons.clear), label: const Text('Clear')),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton.icon(
                  onPressed: pipeline.recordState == RecordState.idle ? onRecord : null,
                  icon: const Icon(Icons.fiber_manual_record),
                  label: const Text('Record sample'),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _preview() {
    final src = source;
    final overlay = CustomPaint(
      painter: LandmarkPainter(pipeline.lastFrame, mirrorX: !src.previewMirrored),
      child: const SizedBox.expand(),
    );
    Widget view;
    if (src is PhoneCameraSource && src.controller != null && src.controller!.value.isInitialized) {
      final c = src.controller!;
      view = AspectRatio(
        aspectRatio: 1 / c.value.aspectRatio,
        child: Stack(fit: StackFit.expand, children: [CameraPreview(c), overlay]),
      );
    } else if (src is MjpegSource && src.lastJpeg != null) {
      final upright = src.rotation % 180 == 0;
      view = AspectRatio(
        aspectRatio: upright ? 4 / 3 : 3 / 4,
        child: Stack(fit: StackFit.expand, children: [
          RotatedBox(
            quarterTurns: src.rotation ~/ 90,
            child: Image.memory(src.lastJpeg!, gaplessPlayback: true, fit: BoxFit.fill),
          ),
          overlay,
        ]),
      );
    } else {
      view = const Text('Waiting for camera...');
    }
    return Stack(
      alignment: Alignment.center,
      children: [
        view,
        if (pipeline.recordState != RecordState.idle) _RecordOverlay(pipeline: pipeline),
      ],
    );
  }
}

class _RecordOverlay extends StatelessWidget {
  const _RecordOverlay({required this.pipeline});

  final TranslationPipeline pipeline;

  @override
  Widget build(BuildContext context) {
    final (text, color) = switch (pipeline.recordState) {
      RecordState.countdown => ('${pipeline.countdown}', Colors.amber),
      RecordState.recording => ('RECORDING', Colors.redAccent),
      RecordState.uploading => ('UPLOADING...', Colors.blueAccent),
      RecordState.saved => (pipeline.recordMessage, Colors.greenAccent),
      RecordState.error => (pipeline.recordMessage, Colors.redAccent),
      RecordState.idle => ('', Colors.transparent),
    };
    return Container(
      padding: const EdgeInsets.all(16),
      color: Colors.black54,
      child: Text(text,
          textAlign: TextAlign.center,
          style: TextStyle(color: color, fontSize: pipeline.recordState == RecordState.countdown ? 72 : 22)),
    );
  }
}
