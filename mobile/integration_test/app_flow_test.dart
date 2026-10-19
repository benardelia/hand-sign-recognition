// End-to-end app flow on a device against a running hub server:
//   (in Handsign/) PORT=5078 python app.py
//   flutter test integration_test/app_flow_test.dart -d <device> [--dart-define=HUB_URL=http://host:5078]
// Syncs the LOW model from /api/models, then feeds MediaPipe's test photo through the
// translation pipeline in place of a camera.
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:integration_test/integration_test.dart';
import 'package:scribesign_glasses/src/frame_sources.dart';
import 'package:scribesign_glasses/src/model_store.dart';
import 'package:scribesign_glasses/src/sign_classifier.dart';
import 'package:scribesign_glasses/src/translation_pipeline.dart';
import 'package:sign_landmarks/sign_landmarks.dart';

const imageUrl = 'https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg';
const hubUrlDefine = String.fromEnvironment('HUB_URL');

/// Replays one JPEG as a 15 fps camera.
class PhotoSource implements FrameSource {
  PhotoSource(this.jpeg);

  final Uint8List jpeg;
  final _frames = StreamController<SourceFrame>.broadcast();
  Timer? _timer;

  @override
  Stream<SourceFrame> get frames => _frames.stream;
  @override
  bool get previewMirrored => false;
  @override
  int get rotation => 0;

  @override
  Future<void> start() async {
    _timer = Timer.periodic(const Duration(milliseconds: 66),
        (_) => _frames.add(SourceFrame(bytes: jpeg, format: FrameFormat.jpeg)));
  }

  @override
  Future<void> stop() async => _timer?.cancel();
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('model sync + live pipeline', (tester) async {
    // The Android emulator reaches the host machine at 10.0.2.2.
    final hub = hubUrlDefine.isNotEmpty
        ? hubUrlDefine
        : (Platform.isAndroid ? 'http://10.0.2.2:5078' : 'http://127.0.0.1:5078');
    final store = ModelStore();
    await SignLandmarks.initialize(modelPath: await store.ensureHolisticModel(), useGpu: false);
    final model = await store.ensureProfile(hub, 'low');
    expect(model.sequenceLength, 15);
    expect(model.labels, isNotEmpty);

    final classifier = await SignClassifier.load(model);
    final sentences = <String>[];
    final pipeline = TranslationPipeline(classifier: classifier, threshold: 0.5, onSentence: sentences.add);
    final jpeg = Uint8List.fromList((await http.get(Uri.parse(imageUrl))).bodyBytes);
    await pipeline.start(PhotoSource(jpeg));
    await tester.runAsync(() => Future<void>.delayed(const Duration(seconds: 6)));
    await pipeline.stop();

    expect(pipeline.error, isEmpty);
    expect(pipeline.lastFrame, isNotNull);
    expect(SignLandmarks.hasHands(pipeline.lastFrame!), isTrue);
    expect(pipeline.fps, greaterThan(1));
    // A full window was classified: the decoder saw a probability distribution.
    expect(pipeline.decoder.confidence, greaterThan(0));
    await classifier.close();
  });
}
