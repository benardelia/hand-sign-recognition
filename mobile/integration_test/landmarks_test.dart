// On-device test of the native sign_landmarks plugin (MediaPipe on Android/iOS).
//   flutter test integration_test -d <device>
// 1. Runs HolisticLandmarker on MediaPipe's holistic test photo and compares key
//    landmarks with the legacy Python Holistic reference (tools/parity_check.py).
// 2. Runs the exported LOW classifier natively and compares it with Python.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_reactive_ble/flutter_reactive_ble.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:integration_test/integration_test.dart';
import 'package:scribesign_glasses/src/model_store.dart';
import 'package:sign_landmarks/sign_landmarks.dart';

import 'classifier_fixture.dart';

const imageUrl = 'https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg';

// Legacy Python Holistic on the mirrored photo (x, y).
const reference = {
  'nose': (0, 0.4928, 0.1517),
  'left shoulder': (44, 0.6050, 0.2755),
  'right shoulder': (48, 0.3710, 0.2741),
  'left wrist': (SignLandmarks.leftHandOffset, 0.7520, 0.4952),
  'right wrist': (SignLandmarks.rightHandOffset, 0.3973, 0.3967),
};

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('native holistic landmarks match the Python reference', (tester) async {
    final modelPath = await ModelStore().ensureHolisticModel();
    final delegate = await SignLandmarks.initialize(modelPath: modelPath, useGpu: false);
    expect(delegate, 'CPU');

    final res = await http.get(Uri.parse(imageUrl));
    expect(res.statusCode, 200);
    final jpeg = Uint8List.fromList(res.bodyBytes);

    late Float32List frame;
    // VIDEO mode: a few frames let the hand and face trackers lock on, like a live stream.
    for (var i = 0; i < 5; i++) {
      frame = await SignLandmarks.detect(bytes: jpeg, format: FrameFormat.jpeg, timestampMs: i * 33);
    }
    expect(frame.length, SignLandmarks.frameSize);
    expect(SignLandmarks.hasHands(frame), isTrue);

    reference.forEach((name, ref) {
      final (offset, x, y) = ref;
      expect(frame[offset], closeTo(x, 0.03), reason: '$name x');
      expect(frame[offset + 1], closeTo(y, 0.03), reason: '$name y');
    });

    // Downscaling to the LOW profile width keeps normalized coordinates.
    final small = await SignLandmarks.detect(
        bytes: jpeg, format: FrameFormat.jpeg, timestampMs: 1000, targetWidth: 320);
    expect(small[SignLandmarks.leftHandOffset], closeTo(0.7520, 0.04));
    await SignLandmarks.close();
  });

  testWidgets('native classifier matches the Python reference output', (tester) async {
    final dir = await Directory.systemTemp.createTemp('classifier');
    final modelFile = File('${dir.path}/low_model.tflite');
    await modelFile.writeAsBytes(base64Decode(lowModelB64));
    final shape = await SignLandmarks.loadClassifier(modelFile.path);
    expect(shape.frames, lowSequenceLength);
    expect(shape.features, SignLandmarks.frameSize);
    expect(shape.classes, lowLabels.length);

    final input = Uint8List.fromList(base64Decode(lowInputB64)).buffer.asFloat32List();
    final probs = await SignLandmarks.classify(input);
    for (var i = 0; i < lowExpected.length; i++) {
      expect(probs[i], closeTo(lowExpected[i], 1e-3), reason: lowLabels[i]);
    }
    await SignLandmarks.closeClassifier();
    await dir.delete(recursive: true);
  });

  // The BLE plugin's protobuf messages run on protobuf-java on Android (see android/build.gradle.kts).
  testWidgets('BLE plugin channel decodes status messages', (tester) async {
    final status = await FlutterReactiveBle().statusStream.first.timeout(const Duration(seconds: 10));
    expect(BleStatus.values, contains(status));
  });
}
