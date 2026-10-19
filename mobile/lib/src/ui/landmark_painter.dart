import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:sign_landmarks/sign_landmarks.dart';

/// Draws a 1662-float holistic frame over the camera preview.
/// Landmarks come from the mirrored, upright image; [mirrorX] flips them back
/// when the preview itself is not mirrored.
class LandmarkPainter extends CustomPainter {
  LandmarkPainter(this.frame, {required this.mirrorX});

  final Float32List? frame;
  final bool mirrorX;

  static const _hand = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
  ];
  static const _pose = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];
  static const _lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];

  @override
  void paint(Canvas canvas, Size size) {
    final f = frame;
    if (f == null) return;
    Offset pt(int offset) => Offset((mirrorX ? 1 - f[offset] : f[offset]) * size.width, f[offset + 1] * size.height);
    bool present(int offset) => f[offset] != 0 || f[offset + 1] != 0;

    final posePaint = Paint()
      ..color = const Color(0xE66366F1)
      ..strokeWidth = 3;
    for (final c in _pose) {
      final a = c[0] * 4, b = c[1] * 4;
      if (present(a) && present(b) && f[a + 3] > 0.4 && f[b + 3] > 0.4) {
        canvas.drawLine(pt(a), pt(b), posePaint);
      }
    }

    final lipsPaint = Paint()
      ..color = const Color(0xAA50FF79)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;
    final lipsPath = Path();
    var started = false;
    for (final i in _lips) {
      final o = SignLandmarks.faceOffset + i * 3;
      if (!present(o)) continue;
      final p = pt(o);
      started ? lipsPath.lineTo(p.dx, p.dy) : lipsPath.moveTo(p.dx, p.dy);
      started = true;
    }
    if (started) canvas.drawPath(lipsPath..close(), lipsPaint);

    for (final (offset, color) in [
      (SignLandmarks.leftHandOffset, const Color(0xFF06B6D4)),
      (SignLandmarks.rightHandOffset, const Color(0xFF10B981)),
    ]) {
      if (!present(offset)) continue;
      final line = Paint()
        ..color = color
        ..strokeWidth = 2.5;
      for (final c in _hand) {
        canvas.drawLine(pt(offset + c[0] * 3), pt(offset + c[1] * 3), line);
      }
      final dot = Paint()..color = color;
      for (var i = 0; i < 21; i++) {
        canvas.drawCircle(pt(offset + i * 3), 3, dot);
      }
    }
  }

  @override
  bool shouldRepaint(LandmarkPainter old) => old.frame != frame || old.mirrorX != mirrorX;
}
