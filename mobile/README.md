# ScribeSign mobile app (Android + iOS)

Flutter app that turns a camera (the phone's, or the smart glasses' MJPEG stream) into
translated sentences, and sends them to the glasses display over Bluetooth.

```
camera frame ──► sign_landmarks plugin (MediaPipe HolisticLandmarker, native)
             ──► 1662-float holistic frame (same layout as the Python/browser collectors)
             ──► sliding window (profile sequence length) ──► model.tflite (native TFLite)
             ──► GlossDecoder ──► sentence ──► screen + glasses (BLE)
```

Everything runs on the phone. The hub server (the Flask app in the repo root) is only
used to download the latest trained model per profile and, optionally, to upload new
training samples recorded with the **Record sample** button.

## Run

```bash
cd mobile
flutter pub get
flutter run            # pick a connected phone
```

First launch downloads the MediaPipe holistic model (~14 MB) and the selected profile's
model from the server set in **Settings** (default `https://handsign.duckdns.org`). After
that the app works offline with the cached models.

Settings:

| Setting | Meaning |
|---|---|
| Quality profile | `LOW` (hands + upper body, 15 fps, ~65 KB model) or `HIGH` (adds lips/eyebrows, 30 fps). Must be trained on the server first (web dashboard → Model Trainer). |
| Camera | Phone camera (front or back) or the glasses' MJPEG stream. |
| Frame rotation | Override if the drawn landmarks do not line up with the person. |
| Confidence threshold | Minimum probability before a sign counts. |
| Server URL / upload token | Hub for model updates and sample uploads (`COLLECT_TOKEN` on the server). |

## Layout

| Path | What |
|---|---|
| `lib/src/translation_pipeline.dart` | frames → landmarks → classifier → sentence; collect mode |
| `lib/src/frame_sources.dart` | phone camera (NV21/BGRA) and glasses MJPEG sources |
| `lib/src/model_store.dart` | downloads/caches the MediaPipe model and `/api/models/<profile>` |
| `lib/src/gloss_decoder.dart` | port of `utils/inference.py` GlossDecoder + `utils/translator.py` |
| `lib/src/glasses_link.dart` | BLE link to `firmware/glasses_esp32s3` |
| `packages/sign_landmarks/` | native plugin: MediaPipe HolisticLandmarker + TFLite classifier (Kotlin / Swift) |

## Tests

```bash
flutter test                                   # decoder, translator, MJPEG parser (host)
flutter test integration_test/landmarks_test.dart -d <device>
#   native landmarks vs the Python reference, native classifier vs Python, BLE channel
PORT=5078 python app.py                        # in the repo root, then:
flutter test integration_test/app_flow_test.dart -d <device>
#   model sync from the hub + full pipeline on MediaPipe's test photo
```

All of these pass on the Android emulator (API 36, arm64) and the iOS 26 simulator.

## Platform notes (why the build files look the way they do)

- **MediaPipe 1.0.0 on both platforms.** The 0.10.33 iOS pods are mis-packaged
  (`Info.plist` says `.a`, the archive ships `.framework`), which breaks linking.
- **iOS classifier uses MediaPipe's TFLite.** MediaPipeTasksCommon force-loads a graph
  library that contains the TFLite C API; any second TFLite runtime (e.g. TensorFlowLiteC)
  fails with duplicate symbols. `SignLandmarksTFLite.h` declares the C functions the
  classifier calls. Android uses LiteRT 1.4.2 (separate `.so`, no clash).
- **Android protobuf.** MediaPipe's HolisticLandmarker calls `Any.Builder.build()` with
  the full-protobuf signature that `protobuf-javalite` lacks (crash at startup), so the app
  excludes `protobuf-javalite` everywhere and ships `protobuf-java` 4.26.1 (a superset).
- **BLE library.** `flutter_reactive_ble` (BSD-3-Clause). `flutter_blue_plus` 2.x requires a
  paid license for commercial use. `reactive_ble_mobile` hardcodes `compileSdkVersion 37`,
  which this Android Gradle Plugin cannot resolve; `android/build.gradle.kts` compiles it
  against 36.
- **Mirroring.** Frames are rotated upright and mirrored before detection, the convention
  of every collector (`config/profiles.json` → `dataset.mirrored`).
