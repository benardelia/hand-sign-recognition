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

## Phase 0 — Profiles and model export (Python only)

- [ ] `config/profiles.yaml` defining `low`, `high`, `experimental`
- [ ] `utils/features.py`: `select_features(frame_1662, profile)` + `normalize()` (wrist- or shoulder-relative)
- [ ] Export `feature_spec.json` (index lists + normalization rules) for the Flutter and JS clients
- [ ] Resample sequences to the profile's sequence length (uses timestamps once collected)
- [ ] `train_model.py --profile <name>` → `Model/<profile>/`
- [ ] TFLite export: int8 for `low`, fp16 for `high`; Keras for `experimental`
- [ ] Python check: TFLite output matches Keras output on held-out samples
- [ ] Keep the current `Model/hand_model.h5` path working until Phase 2 replaces it

**Done when:** one command trains each profile from existing data and reports per-profile accuracy and model size.

## Phase 1 — Web collector works remotely

- [ ] Replace the JPEG streaming loop in `static/js/main.js` with `@mediapipe/tasks-vision` `HolisticLandmarker` running in the browser
- [ ] Draw landmarks client-side on a canvas, and run the countdown and recording in the browser
- [ ] `POST /api/samples`: validate shape (N×1662), label and metadata, then save `.npy` + `.json`
- [ ] Metadata per sample: timestamps, fps, resolution, source (`web`/`phone`/`pc`), user agent, contributor id
- [ ] Per-session state, so several contributors can record at once
- [ ] Gate the OpenCV/server camera path behind `LOCAL_CAMERA=1`, and drop MediaPipe/OpenCV from the production image
- [ ] **Parity test:** record the same clip with Python Holistic and JS Holistic, then compare feature order and value ranges (face: 478 vs 468 points, left/right hand handedness)
- [ ] Confirm the production deployment serves HTTPS (browsers block the camera without it)

**Done when:** a contributor on a phone browser, on the remote server, records 10 samples and they appear in the dataset with metadata.

## Phase 2 — Model serving API

- [ ] `POST /api/predict`: takes a landmark sequence + profile and returns label + confidence (live preview in the web app)
- [ ] `GET /api/models/<profile>`: latest `model.tflite`, `labels.txt`, `profile.json`, `feature_spec.json` and a version hash
- [ ] Train from the web UI with a profile selector
- [ ] Dataset dashboard: samples per label per source, and flags for labels with too few samples

## Phase 3 — Flutter app (Android + iOS)

- [ ] Flutter project `mobile/` with a federated plugin `sign_landmarks`:
  - [ ] Android: Kotlin wrapper around MediaPipe Tasks `HolisticLandmarker` (live-stream mode)
  - [ ] iOS: Swift wrapper around `MediaPipeTasksVision` `HolisticLandmarker` (CocoaPods)
  - [ ] One Dart API returning a 1662-float frame in the same order as `utils/holistic_utils.py`
- [ ] Classifier with `flutter_litert` (Interpreter API), driven by `profile.json`
- [ ] Port `feature_spec.json` selection and normalization to Dart, with unit tests against Python fixtures
- [ ] Gloss → sentence logic ported from `utils/translator.py`
- [ ] Settings screen: profile (low/high), threshold, model update check
- [ ] Phone camera as input first (no glasses needed for testing)
- [ ] Optional: "collect" mode posting to `/api/samples`, so phones add training data
- [ ] Performance budget: ≥15 fps end-to-end on a mid-range Android phone and on an iPhone

## Phase 4 — Glasses hardware

- [ ] Choose the camera board (ESP32-S3 + OV2640/OV5640, or an off-the-shelf camera module)
- [ ] MJPEG over Wi-Fi from the glasses to the phone, consumed by the Flutter app
- [ ] Text back to the glasses display over BLE (`flutter_blue_plus`)
- [ ] Measure latency and battery life
- [ ] Camera angle: the wearer sees the *other* person signing, so collect training data from that viewpoint

## Phase 5 — Experimental track (PC)

- [ ] `realtime_holistic.py --profile experimental`
- [ ] Try Transformer / 1D-CNN architectures (see the Kaggle ISLR references)
- [ ] Continuous signing (sentences instead of isolated signs)

---

## Housekeeping found during planning

- [ ] `/api/npy_data` (`app.py`) loads any client-supplied path. Restrict it to `DATA_PATH`, as `/api/delete_file` does
- [ ] `.env` is tracked in git. Untrack it, add it to `.gitignore`, and rotate its secrets
- [ ] Remove committed `__pycache__/` files

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-10-19 | Arduino dropped as the inference target | Python, TensorFlow and MediaPipe Holistic need far more RAM and compute than any Arduino-class board |
| 2026-10-19 | Option 1: glasses capture, phone processes | Matches how commercial smart glasses work, and reuses the current pipeline |
| 2026-10-20 | Flutter for Android + iOS | One codebase. MediaPipe has no official Flutter plugin, so landmarks go through a small native plugin per platform |
| 2026-10-20 | Store full 1662 features; profiles select subsets | Switch low/high/experimental without re-collecting data |

## Risks / open questions

- Python Holistic (legacy) vs Tasks Holistic (JS/Android/iOS) may differ in landmark count, order or handedness. The Phase 1 parity test must pass before mixing data from different sources.
- Community Flutter MediaPipe packages are Android-only or unofficial, so the plan uses our own thin native plugin.
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
