# sign_landmarks

Flutter plugin wrapping MediaPipe Tasks `HolisticLandmarker` (Android and iOS, MediaPipe 1.0.0).

`SignLandmarks.detect()` takes a camera frame (JPEG, Android NV21 or iOS BGRA8888), rotates it upright,
mirrors it (selfie view) and returns the 1662-float holistic frame used everywhere in this project:

| range | content |
|---|---|
| 0-131 | pose: 33 x (x, y, z, visibility) |
| 132-1535 | face: 468 x (x, y, z) |
| 1536-1598 | left hand: 21 x (x, y, z) |
| 1599-1661 | right hand: 21 x (x, y, z) |

The model file (`holistic_landmarker.task`) is passed by path to `SignLandmarks.initialize()`.
