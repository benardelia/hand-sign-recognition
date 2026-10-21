# Smart Glasses Translator — Tracker

Goal: real-time sign language translation shown on smart glasses. The glasses only capture video and display text. A **Flutter app (Android + iOS)** on the phone runs landmark extraction and the classifier. The **Flask web app** is the hub for data collection and training.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped

---

## Architecture

```
 Glasses (ESP32-S3 cam + display)
   │  MJPEG over Wi-Fi                ▲ translated text over BLE
   ▼                                  │
 Flutter app (Android / iOS)
   ├─ native plugin: MediaPipe Tasks HolisticLandmarker (Kotlin / Swift)
   ├─ feature selection (feature_spec.json for the active profile)
   ├─ classifier: model.tflite via flutter_litert
   └─ gloss → sentence
            │  GET /api/models/<profile>    (download latest model)
            ▼
 Flask hub (server)
   ├─ browser collector: MediaPipe JS in the browser → POST /api/samples
   ├─ dataset: Data/Holistic_Landmarks/<LABEL>/*.npy + *.json metadata
   ├─ train: train_model.py --profile low|high|experimental
   └─ Model/<profile>/{model.tflite, labels.txt, profile.json, feature_spec.json}
```

Core rules:
1. **Always store the full 1662-feature holistic frame.** Profiles are subsets chosen at training and inference time, so switching quality never requires re-collecting data.
2. **Images never leave the device.** The browser and the phone extract landmarks locally and send only numbers.
3. **Read `profile.json` at runtime instead of hard-coding.** Switching profile means changing one setting.

---

## Quality profiles

| | `low` | `high` | `experimental` |
|---|---|---|---|
| Target | Glasses → mid-range phone | Flagship phone | Powerful PC / GPU |
| Camera | 320×240 or 480×360 @ 15 fps | 640×480 @ 30 fps | 1280×720 @ 30+ fps |
| Features / frame | ~150 (both hands + 8 upper-body pose points) | ~400 (hands + upper-body pose + ~40 lip/brow points) | 1662 (full holistic) |
| Sequence length | 15 | 30 | 30–60 |
| Model | Small GRU, int8 TFLite (~50–100 KB) | GRU/1D-CNN, fp16 TFLite (~0.5 MB) | LSTM/Transformer, Keras, PC only |
| Runs where | Phone | Phone | `realtime_holistic.py` / Flask local mode |

Final feature counts are recorded in `config/profiles.yaml` once Phase 0 is implemented.

---

## Phase 0 — Profiles and model export (Python only) ✅

- [x] `config/profiles.json` defining `low`, `high`, `experimental`
- [x] `utils/features.py`: `select_features(frame_1662, spec)` + shoulder-centred normalization; pose z masked (unreliable across MediaPipe graphs)
- [x] Selection + normalization run **inside the model** (`utils/tf_layers.py`), so clients only feed raw `(T, 1662)` frames; `feature_spec.json` exported for reference
- [x] Resample sequences to the profile's sequence length (linear, never blending a detected and a missing landmark)
- [~] Timestamp-aware resampling: timestamps are now stored with every sample; training still resamples by frame index
- [x] `train_model.py --profile <name|all>` → `Model/<profile>/`
- [x] TFLite export (builtin ops only): dynamic int8 for `low` (~65 KB), float16 for `high` (~0.7 MB); Keras for `experimental`
- [x] TFLite vs Keras check during training (max diff ~1e-5)
- [x] Legacy `Model/hand_model.h5` still loads when a profile has no model yet

## Phase 1 — Web collector works remotely ✅

- [x] Browser-side MediaPipe `HolisticLandmarker` (`static/js/pipeline.js`), mirrored, at the profile's resolution/fps
- [x] Client-side drawing, countdown and recording; slow devices extend the recording to ≥ 8 frames
- [x] `POST /api/samples` with validation and `.json` metadata (source, fps, timestamps, resolution, device)
- [x] Per-session live translation (`/api/stream/frames`, binary float32), many contributors at once
- [x] Server camera behind `LOCAL_CAMERA=1`; production image has no OpenCV/MediaPipe
- [x] **Parity test** (`tools/parity_check.py` + headless Chrome): browser landmarks match legacy Python within 0.3% (x/y), handedness consistent; face missed more often by the Tasks graph → face-dropout augmentation
- [x] Optional `ADMIN_TOKEN` / `COLLECT_TOKEN`
- [ ] Confirm production serves the dashboard over HTTPS (needed for the browser camera)

## Phase 2 — Model serving API ✅

- [x] `POST /api/predict`
- [x] `GET /api/models` and `/api/models/<profile>/<file>` (with version + sha256)
- [x] Train from the web UI with a profile selector
- [x] Dataset view shows samples per label and per source (web / phone / glasses / pc)
- [x] `tools/import_media.py`: import downloaded videos / photos (folder = label) and Kaggle ISLR landmark files, source `import`, duplicate-safe

## Phase 3 — Flutter app (Android + iOS) ✅ (needs real-device field test)

- [x] `mobile/` app + `sign_landmarks` plugin: MediaPipe HolisticLandmarker (Kotlin / Swift), JPEG / NV21 / BGRA input, rotation, mirroring, downscaling
- [x] Native sign classifier (LiteRT on Android; MediaPipe's own TFLite C API on iOS)
- [x] Model sync from the hub, cached offline; LOW / HIGH switch in Settings
- [x] Gloss decoder + translator ported to Dart (unit tests)
- [x] Phone camera and glasses MJPEG sources; collect mode uploads samples
- [x] Integration tests pass on Android emulator + iOS simulator (landmarks vs Python, classifier vs Python, full flow with hub sync)
- [ ] Field test on real phones: camera rotation per device, ≥ 15 fps end-to-end on a mid-range Android and an iPhone
- [ ] App icon, store signing, release builds

## Phase 4 — Glasses hardware (firmware ready, hardware pending)

- [x] Board choice: Seeed XIAO ESP32S3 Sense (+ SSD1306 OLED); ESP32-S3-EYE pin map included
- [x] Firmware: MJPEG over Wi-Fi (AP or station), BLE text + stream-URL characteristics (`firmware/glasses_esp32s3`)
- [x] App ↔ glasses BLE link (`flutter_reactive_ble`)
- [ ] Build the prototype; measure latency and battery life
- [ ] Collect training data from the glasses' viewpoint (Record sample with Camera = Glasses)

## Phase 5 — Experimental track (PC)

- [x] `realtime_holistic.py --profile experimental`
- [x] CNN + Transformer architecture for `experimental`
- [ ] Continuous signing (sentences instead of isolated signs) — needs a much larger dataset

---

## Housekeeping ✅

- [x] `/api/npy_data` restricted to the dataset folder
- [x] `.env` untracked
- [x] `__pycache__/` untracked

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-10-19 | Arduino dropped as the inference target | Python, TensorFlow and MediaPipe Holistic need far more RAM and compute than any Arduino-class board |
| 2026-10-19 | Option 1: glasses capture, phone processes | Matches how commercial smart glasses work, and reuses the current pipeline |
| 2026-10-20 | Flutter for Android + iOS | One codebase. MediaPipe has no official Flutter plugin, so landmarks go through a small native plugin per platform |
| 2026-10-20 | Store full 1662 features; profiles select subsets | Switch low/high/experimental without re-collecting data |
| 2026-10-20 | Feature selection/normalization inside the model graph | Browser, phone and server feed identical raw frames; no per-language reimplementation |
| 2026-10-20 | Mask pose z; face-dropout augmentation | Parity check: pose z differs strongly between MediaPipe graphs; Tasks graph misses faces more often |
| 2026-10-20 | MediaPipe 1.0.0 on Android and iOS | 0.10.33 iOS pods are mis-packaged (.a vs .framework) |
| 2026-10-20 | Classifier runs natively in the plugin (not flutter_litert) | A second TFLite runtime clashes with MediaPipe's on iOS (duplicate symbols) |
| 2026-10-20 | protobuf-java instead of protobuf-javalite on Android | MediaPipe Holistic calls the full-protobuf `Any.Builder.build()` (crash otherwise) |
| 2026-10-20 | flutter_reactive_ble (BSD-3) for BLE | flutter_blue_plus 2.x needs a paid license for commercial use |
| 2026-10-20 | XIAO ESP32S3 Sense + SSD1306 OLED for the prototype | Camera + PSRAM in 21 x 17.5 mm; cheap, Arduino-friendly |
| 2026-10-22 | Imports never guess whether a file was mirrored | MediaPipe names body sides by appearance, so mirroring is invisible in landmarks; a wrong guess only swaps the dominant hand, which mirror augmentation covers |

## Risks / open questions

- ~~Python Holistic (legacy) vs Tasks Holistic may differ~~ — checked: x/y within 0.3%, handedness consistent; pose z masked.
- Only ~28 holistic samples exist today: accuracy numbers are not meaningful until each sign has 15-20+ samples from several people.
- Data collected from a laptop webcam (signer facing the camera) differs from the glasses' view (signer opposite the wearer, possibly at an angle).
- On-device fps on low-end phones with full Holistic. Fallback: `low` uses Hand + Pose landmarkers only, without the face.

---

## References

**Closest existing projects**
- Sign language translation glasses on a Raspberry Pi Zero 2 W — https://www.raspberrypi.com/news/sign-language-translation-glasses/
- `vision_flow`, Flutter plugin for sign/gesture recognition from MediaPipe landmarks (Android only) — https://pub.dev/packages/vision_flow
- Kaggle: Google Isolated Sign Language Recognition (TFLite model on MediaPipe Holistic landmarks; same problem as ours) — 1st-place solution repo: https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution
- Another ISLR Kaggle solution — https://github.com/JosephZahar/Google-Isolated-Sign-Language-Recognition-Kaggle
- Landmark-subset results (fingerspelling winner used 130 landmarks: hands, arms, lips/nose/eyes) — https://habr.com/en/articles/795109
- Real Time Sign Language Detection with MediaPipe — https://github.com/swojas/RTSLD

**SDKs / packages**
- MediaPipe Holistic Landmarker (overview) — https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
- Holistic Landmarker for iOS — https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker/ios
- `flutter_litert` (successor of `tflite_flutter`; Android, iOS, desktop, web) — https://pub.dev/packages/flutter_litert
- `hand_landmarker` (Flutter, Android only, MediaPipe via JNI) — https://pub.dev/packages/hand_landmarker
- `hand_detection` (Flutter, MediaPipe hand models via TFLite, multi-platform) — https://pub.dev/documentation/hand_detection/4.3.1/
- Known issue: left/right body point separation in the Tasks API — https://github.com/google/mediapipe/issues/4895
