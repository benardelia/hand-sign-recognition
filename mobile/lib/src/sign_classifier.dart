import 'dart:typed_data';

import 'package:sign_landmarks/sign_landmarks.dart';

import 'model_store.dart';

/// Runs `Model/<profile>/model.tflite` natively (sign_landmarks plugin). Input:
/// the last `sequenceLength` raw holistic frames (1662 floats each); feature
/// selection and normalization happen inside the model, exactly as on the server.
class SignClassifier {
  SignClassifier._(this.model) : _input = Float32List(model.sequenceLength * frameSize);

  static const frameSize = 1662;

  final ProfileModel model;
  final Float32List _input;

  static Future<SignClassifier> load(ProfileModel model) async {
    final shape = await SignLandmarks.loadClassifier(model.tflitePath);
    if (shape.frames != model.sequenceLength || shape.features != frameSize || shape.classes != model.labels.length) {
      await SignLandmarks.closeClassifier();
      throw StateError('Model shape $shape does not match profile '
          '(${model.sequenceLength} frames, $frameSize features, ${model.labels.length} labels)');
    }
    return SignClassifier._(model);
  }

  /// [window] must hold exactly `sequenceLength` frames, oldest first.
  Future<Float32List> predict(List<Float32List> window) {
    for (var i = 0; i < window.length; i++) {
      _input.setRange(i * frameSize, (i + 1) * frameSize, window[i]);
    }
    return SignLandmarks.classify(_input);
  }

  Future<void> close() => SignLandmarks.closeClassifier();
}
