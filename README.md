# Hand Sign Recognition

A real-time sign language translation system: MediaPipe holistic landmarks, small
sequence models in three quality profiles, a web hub for data collection and training,
and a Flutter app that runs the translator for smart glasses.

```
glasses camera ─MJPEG─► phone app (MediaPipe + TFLite) ─BLE─► glasses display
                              ▲ model updates          │ new samples
                              └──── web hub (Flask) ◄──┘ ◄── browser data collection
```

## Project Overview
This project is a high-performance **Landmark-based** recognition system. It has evolved from simple hand tracking to a full **Holistic** system that extracts coordinates for the face, body (pose), and both hands. This allows the AI to recognize complex signs that rely on facial expressions and body posture, translating them into natural English sentences.

## Features
- **Holistic Tracking**: Uses MediaPipe Holistic to capture 1662 landmarks (Face Mesh, Pose, and Both Hands).
- **Quality Profiles**: `low` / `high` (phone, TFLite) and `experimental` (PC, CNN + Transformer) trained from the same data; switch with one setting (`config/profiles.json`).
- **Browser Data Collection**: landmarks are extracted in the browser, so the hosted dashboard collects data from any device; only numbers are uploaded.
- **Mobile App + Glasses**: Flutter app (Android/iOS) with native MediaPipe and TFLite; ESP32-S3 glasses firmware streams the camera and shows the text.
- **Smart Sentence Reconstruction**: Automatically strings together detected keywords into grammatically correct English sentences.
- **Advanced Logging**: Color-coded console logs with clickable file links for easier debugging.
- **Real-time Translation**: Live webcam feed with overhead predictions and a sentence display bar.
- **Visual Verification**: Built-in tools to replay and inspect collected landmark data.

## Installation

### Prerequisites
- Python 3.10+
- Webcam

### Setup
1. Clone the repository:
   ```bash
    git clone git@github.com:benardelia/hand-sign-recognition.git
    cd hand-sign-recognition
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements-local.txt   # local machine: server + webcam tools
   # pip install -r requirements.txt       # server only (what the Docker image uses)
   ```

## Workflow

### 1. Data Collection
Collect data for each sign you want the AI to learn:
- **Hand-Only (Legacy)**: see [legacy/README.md](legacy/README.md).
- **Holistic Tracking (Advanced)**: Run `python scripts/collect_holistic.py` (or use the web dashboard). Press **'s'** to record a 30-frame sequence including your face, pose, and hands.

#### Importing downloaded data
`tools/import_media.py` turns downloaded material into training samples (source `import`):
```bash
# One folder per sign; the folder name is the label
python tools/import_media.py media downloads/ --dataset-name my-dictionary --license "CC BY 4.0"
# Kaggle "Isolated Sign Language Recognition" landmark files (American Sign Language)
python tools/import_media.py kaggle path/to/asl-signs --labels hello thankyou --limit-per-label 50
```
- **Videos** are trimmed to the part where hands are visible (max 2 s, `--max-seconds`).
- **Photos** only help signs that don't move (handshapes, fingerspelling): one frame is
  repeated to a recording's length.
- Clips without hands are skipped, and re-running skips files already imported. `--dry-run`
  shows what would be added.
- Frames are mirrored like the collectors (use `--no-flip` for clips that are already
  mirrored).
- Check the licence of anything you download. Use data from the sign language your users
  sign.

### 2. Verification
Check your data quality before training:
- **Numerical View**: `python tools/preview_npy.py <path_to_file.npy>`
- **Skeletal Animation**: `python tools/visualize_landmarks.py <path_to_file.npy>`

### 3. Training
Every quality profile trains on the same holistic data (`Data/Holistic_Landmarks`):
```bash
python ml/train_model.py --profile low         # or high, experimental, all
```
Output goes to `Model/<profile>/`: `model.keras`, `model.tflite` (low/high), `labels.txt`,
`profile.json` and `feature_spec.json`. Feature selection and normalization are part of the
model, so every client feeds the raw `(T, 1662)` landmark sequence.

| Profile | Runs on | Camera | Landmarks | Model |
|---|---|---|---|---|
| `low` | glasses + mid-range phone | 480x360 @ 15 fps | hands + 7 upper-body points | small CNN, ~65 KB TFLite |
| `high` | flagship phone | 640x480 @ 30 fps | + lips, eyebrows, upper body | residual CNN, ~0.7 MB TFLite |
| `experimental` | PC | 1280x720 @ 30 fps | full holistic | CNN + Transformer (Keras) |

### 4. Real-time Translation (CLI)
Run the desktop-based translation system to see the AI in action:
- **Holistic**: `python scripts/realtime_holistic.py --profile experimental` (any profile works)

**Controls:**
- **'C'**: Clear the current sentence.
- **'Q'**: Quit the application.

### 5. Interactive Web Dashboard (Flask)
A fully featured interactive web-based dashboard is available for data collection, live translation, model training, and sequence playback:

```bash
# Start the web dashboard (uses virtual environment's Python)
./venv/bin/python app.py
```
Open **http://127.0.0.1:5001** (or the fallback port logged on startup) in your web browser.

Landmarks are extracted **in the browser** (MediaPipe Tasks), so the dashboard works the
same on localhost and on the hosted server (HTTPS required for the camera). Set
`LOCAL_CAMERA=1` to use the server machine's own webcam instead.

For public deployments set `ADMIN_TOKEN` (protects training and deleting) and optionally
`COLLECT_TOKEN` (protects uploads) in `.env`; the dashboard asks for the token once.

**Web Dashboard Features:**
- **Live Translator**: Real-time sign-to-speech translation overlay with audio Text-To-Speech (TTS), reconstruction logs, and copy-to-clipboard.
- **Dataset Builder**: A dedicated capture panel with a live camera preview. You can record consecutive sequences under a single label without re-typing.
- **Quality profile selector**: switches camera resolution, frame rate and model.
- **Model Trainer**: Pick a profile, optionally override epochs/batch size, and watch real-time graphs and console output.
- **Landmark Player**: Browse your saved `.npy` files and replay them in a 2D skeletal animation viewer.

**Stopping the Dashboard:**
- To stop the server: Press **`Ctrl + C`** in your terminal.
- **Automatic Port Recovery**: If port `5001` is blocked or in use, the server will **automatically fall back** to the next available port (e.g., `5002`).
- If you need to manually find and terminate an orphaned background server running on port `5001`:
  ```bash
  # 1. Find the Process ID (PID)
  lsof -i :5001
  
  # 2. Terminate the process using the PID found
  kill <PID>
  ```

### 6. Docker Deployment

#### Local Development with Docker Compose
```bash
# 1. Copy environment variables file
cp .env.example .env

# 2. Build and start the container in development mode
docker compose up --build -d

# 3. View container logs
docker compose logs -f
```

#### Production Deployment (with Reverse Proxy / SSL)
```bash
# Run production compose configuration
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 7. Mobile App and Smart Glasses
- `mobile/`: Flutter app for Android and iOS. See [mobile/README.md](mobile/README.md).
- `firmware/glasses_esp32s3/`: ESP32-S3 glasses firmware. See [its README](firmware/glasses_esp32s3/README.md).

### 8. Tests
```bash
python -m unittest discover tests         # features, dataset API, sessions
python tools/parity_check.py --help       # compare legacy vs Tasks/browser landmarks
cd mobile && flutter test                 # app logic; integration_test/ runs on devices
```

## Project Roadmap
See [ROADMAP.md](ROADMAP.md) for details:
- **Phase 1**: Landmark Collection (COMPLETED)
- **Phase 2**: Sequence Modeling (COMPLETED)
- **Phase 3**: Sentence Reconstruction (COMPLETED)
- **Phase 4**: Holistic Body/Face Tracking (COMPLETED)
- **Phase 5**: Smart Glasses + Mobile ([tracker](docs/SMART_GLASSES_TRACKER.md))

## Project Structure
Run every command from the repository root (`Data/`, `Model/` and `config/` are relative to it).

```
app.py                  web hub: dashboard, collection / prediction / model APIs, training
server/local_camera.py  server-webcam mode (LOCAL_CAMERA=1)
templates/ static/      dashboard UI; static/js/pipeline.js extracts landmarks in the browser
config/profiles.json    quality profiles and dataset conventions
ml/train_model.py       trains and exports a profile (--profile low|high|experimental|all)
scripts/                desktop tools that use the webcam
  collect_holistic.py     record holistic sequences
  realtime_holistic.py    live translation (--profile)
tools/                  data utilities
  import_media.py         import downloaded videos / photos / Kaggle landmarks
  parity_check.py         compare landmarks between MediaPipe versions
  preview_npy.py          print a recorded sequence
  visualize_landmarks.py  replay a recorded sequence as a skeleton
utils/                  shared code
  holistic_utils.py       1662-value holistic frame extraction
  features.py, tf_layers.py  profile feature selection + normalization (numpy / in-model)
  inference.py            profile-aware model loading, gloss decoder
  dataset.py, sessions.py sample storage, per-client live sessions
  translator.py           gloss-to-text sentences
  logger_config.py        logging
tests/                  python -m unittest discover tests
mobile/                 Flutter app + sign_landmarks native plugin
firmware/               glasses firmware (ESP32-S3)
legacy/                 v1 hand-only pipeline (63 features, cvzone), kept for reference
docs/                   smart-glasses tracker
Data/  Model/           datasets and trained models (not in git)
```

## Acknowledgments
- [cvzone](https://github.com/cvzone/cvzone) for simplified tracking utilities.
- [MediaPipe](https://mediapipe.dev/) for state-of-the-art coordinate extraction.
