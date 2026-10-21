# Legacy: v1 hand-only pipeline

The first version of the project: one hand, 63 landmark values (21 points x 3), tracked
with cvzone. Kept for reference; the current pipeline is holistic (1662 values, quality
profiles) — see the main README.

| File | What it did |
|---|---|
| `collect_sign_dataset.py` | record static hand landmarks into `Data/Static/<label>/` |
| `collect_landmarks.py` | record 30-frame hand motion into `Data/Landmarks/<label>/` |
| `realtime_inference.py` | live translation with a hand-only (63-feature) model |
| `sign_detector.py` | image classifier with cvzone's `Classifier` (Teachable Machine style) |
| `normalization.py` | wrist-relative scaling and mirroring of 63-value hand frames |

Run them from the repository root, e.g. `python legacy/collect_landmarks.py`.

They no longer work with the current models: `ml/train_model.py` trains on holistic data
only, and `sign_detector.py` expects an image classifier, while `Model/hand_model.h5` is
an LSTM over landmarks.
