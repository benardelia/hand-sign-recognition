/// Port of utils/inference.py GlossDecoder and utils/translator.py, so the app
/// builds sentences exactly like the web dashboard.
library;

import 'dart:typed_data';

class GlossTranslator {
  static const Map<String, String> phrases = {
    'I WANT WATER': 'I would like some water.',
    'I WANT EAT': 'I am hungry, I want to eat.',
    'HELLO HOW YOU': 'Hello, how are you?',
    'THANK YOU': 'Thank you very much.',
    'NAME ME': 'My name is...',
    'A B C': 'Alphabet start.',
  };

  static String translate(List<String> glosses) {
    if (glosses.isEmpty) return '';
    final key = glosses.join(' ').toUpperCase();
    final phrase = phrases[key];
    if (phrase != null) return phrase;
    final joined = glosses.join(' ');
    var sentence = joined[0].toUpperCase() + joined.substring(1).toLowerCase();
    if (!sentence.endsWith('.') && !sentence.endsWith('?') && !sentence.endsWith('!')) sentence += '.';
    return sentence;
  }
}

class GlossDecoder {
  GlossDecoder(this.labels, {this.threshold = 0.85, this.stabilityFrames = 5, this.maxGlosses = 5}) {
    reset();
  }

  final List<String> labels;
  double threshold;
  final int stabilityFrames;
  final int maxGlosses;

  List<String> glossBuffer = [];
  String sentence = '';
  String lastPrediction = '';
  double confidence = 0;
  int _counter = 0;

  void reset([String initial = 'Start signing...']) {
    glossBuffer = [];
    sentence = initial;
    lastPrediction = '';
    confidence = 0;
    _counter = 0;
  }

  /// Returns the label of this window when it is above the threshold, else null.
  /// The sentence changes only after the same label wins [stabilityFrames] windows in a row.
  String? update(Float32List probs) {
    var index = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[index]) index = i;
    }
    confidence = probs[index];
    if (confidence <= threshold) return null;
    final word = labels[index];
    if (word == lastPrediction) {
      _counter++;
    } else {
      _counter = 0;
      lastPrediction = word;
    }
    if (_counter == stabilityFrames) {
      if (glossBuffer.isEmpty || glossBuffer.last != word) {
        glossBuffer = [...glossBuffer, word];
        if (glossBuffer.length > maxGlosses) glossBuffer = glossBuffer.sublist(glossBuffer.length - maxGlosses);
        sentence = GlossTranslator.translate(glossBuffer);
      }
      _counter = 0;
    }
    return word;
  }
}
