import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:scribesign_glasses/src/frame_sources.dart';
import 'package:scribesign_glasses/src/gloss_decoder.dart';

Float32List probs(List<double> v) => Float32List.fromList(v);

void main() {
  group('GlossTranslator (matches utils/translator.py)', () {
    test('known phrase', () => expect(GlossTranslator.translate(['THANK', 'YOU']), 'Thank you very much.'));
    test('fallback capitalizes like Python str.capitalize', () {
      expect(GlossTranslator.translate(['HELLO', 'FRIEND']), 'Hello friend.');
    });
    test('empty', () => expect(GlossTranslator.translate([]), ''));
  });

  group('GlossDecoder (matches utils/inference.py)', () {
    test('needs stable predictions before adding a gloss', () {
      final d = GlossDecoder(['A', 'B'], stabilityFrames: 2);
      for (var i = 0; i < 2; i++) {
        d.update(probs([0.95, 0.05]));
      }
      expect(d.glossBuffer, isEmpty);
      d.update(probs([0.95, 0.05]));
      expect(d.glossBuffer, ['A']);
      expect(d.sentence, 'A.');
    });
    test('ignores low confidence', () {
      final d = GlossDecoder(['A', 'B']);
      expect(d.update(probs([0.6, 0.4])), isNull);
      expect(d.confidence, closeTo(0.6, 1e-6));
    });
    test('does not repeat the same gloss', () {
      final d = GlossDecoder(['A', 'B'], stabilityFrames: 0);
      d.update(probs([0.99, 0.01]));
      d.update(probs([0.99, 0.01]));
      expect(d.glossBuffer, ['A']);
    });
  });

  group('MjpegParser', () {
    final jpegA = Uint8List.fromList([0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9]);
    final jpegB = Uint8List.fromList([0xFF, 0xD8, 9, 0xFF, 0x00, 8, 0xFF, 0xD9]);
    List<int> part(Uint8List jpeg) => [
          ...ascii.encode('--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.length}\r\n\r\n'),
          ...jpeg,
          ...ascii.encode('\r\n'),
        ];

    test('extracts frames from one chunk', () {
      final frames = MjpegParser().add([...part(jpegA), ...part(jpegB)]);
      expect(frames, [jpegA, jpegB]);
    });
    test('handles frames and markers split across chunks', () {
      final stream = [...part(jpegA), ...part(jpegB)];
      final parser = MjpegParser();
      final frames = <Uint8List>[];
      for (var i = 0; i < stream.length; i += 3) {
        frames.addAll(parser.add(stream.sublist(i, i + 3 > stream.length ? stream.length : i + 3)));
      }
      expect(frames, [jpegA, jpegB]);
    });
  });
}
