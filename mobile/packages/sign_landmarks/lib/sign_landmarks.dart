/// MediaPipe HolisticLandmarker for Android (Kotlin) and iOS (Swift).
///
/// Produces the same 1662-float frame as the Python collectors
/// (utils/holistic_utils.py) and the browser (static/js/pipeline.js):
///   pose 33 x (x, y, z, visibility) | face 468 x (x, y, z) | left hand 21 x 3 | right hand 21 x 3
/// Missing parts are zeros. Images are rotated upright and mirrored (selfie
/// view) natively before detection, which is the convention for all clients.
library;

import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Pixel layout of the bytes passed to [SignLandmarks.detect].
enum FrameFormat {
  /// Encoded JPEG (e.g. one frame of the glasses' MJPEG stream).
  jpeg,

  /// Android camera NV21 (single plane: Y followed by interleaved VU).
  nv21,

  /// iOS camera BGRA8888 (needs [SignLandmarks.detect]'s `bytesPerRow`).
  bgra8888,
}

class SignLandmarks {
  SignLandmarks._();

  static const frameSize = 1662;
  static const poseOffset = 0;
  static const faceOffset = 132;
  static const leftHandOffset = 1536;
  static const rightHandOffset = 1599;

  static const MethodChannel _channel = MethodChannel('sign_landmarks');

  /// Loads the holistic_landmarker.task model from [modelPath] (a file on disk).
  /// Returns the delegate in use ("GPU" or "CPU"); falls back to CPU when the GPU fails.
  static Future<String> initialize({required String modelPath, bool useGpu = true}) async {
    final delegate = await _channel.invokeMethod<String>('initialize', {
      'modelPath': modelPath,
      'useGpu': useGpu,
    });
    return delegate ?? 'CPU';
  }

  /// Runs holistic detection on one video frame.
  ///
  /// [rotationDegrees] rotates the raw image clockwise to make it upright
  /// (for phone cameras: the sensor orientation in portrait). [targetWidth]
  /// downscales the upright image before detection (the profile resolution).
  /// [timestampMs] must increase between calls (VIDEO running mode).
  static Future<Float32List> detect({
    required Uint8List bytes,
    required FrameFormat format,
    required int timestampMs,
    int width = 0,
    int height = 0,
    int bytesPerRow = 0,
    int rotationDegrees = 0,
    bool mirror = true,
    int targetWidth = 0,
  }) async {
    final result = await _channel.invokeMethod<Float32List>('detect', {
      'bytes': bytes,
      'format': format.name,
      'width': width,
      'height': height,
      'bytesPerRow': bytesPerRow,
      'rotation': rotationDegrees,
      'mirror': mirror,
      'targetWidth': targetWidth,
      'timestampMs': timestampMs,
    });
    if (result == null || result.length != frameSize) {
      throw StateError('sign_landmarks: invalid frame from native side');
    }
    return result;
  }

  static Future<void> close() => _channel.invokeMethod('close');

  /// Loads a sign classifier (.tflite exported by train_model.py) into the native
  /// TFLite runtime: LiteRT on Android, and on iOS the TFLite runtime MediaPipe
  /// already contains (a second TFLite library would clash with it at link time).
  /// Returns the model's input frames, input features and number of classes.
  static Future<({int frames, int features, int classes})> loadClassifier(String modelPath) async {
    final shape = await _channel.invokeListMethod<int>('loadClassifier', {'modelPath': modelPath});
    if (shape == null || shape.length != 3) throw StateError('sign_landmarks: invalid classifier shape');
    return (frames: shape[0], features: shape[1], classes: shape[2]);
  }

  /// Runs the loaded classifier on `frames * features` floats (oldest frame first).
  static Future<Float32List> classify(Float32List input) async {
    final result = await _channel.invokeMethod<Float32List>('classify', {'input': input});
    if (result == null) throw StateError('sign_landmarks: classifier returned nothing');
    return result;
  }

  static Future<void> closeClassifier() => _channel.invokeMethod('closeClassifier');

  static bool hasHands(Float32List frame) {
    for (var i = leftHandOffset; i < frameSize; i++) {
      if (frame[i] != 0) return true;
    }
    return false;
  }
}
