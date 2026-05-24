# Hand Sign Recognition

A real-time hand sign detection and translation system using Python, OpenCV, and MediaPipe.

## Project Overview
This project is a high-performance **Landmark-based** recognition system. It has evolved from simple hand tracking to a full **Holistic** system that extracts coordinates for the face, body (pose), and both hands. This allows the AI to recognize complex signs that rely on facial expressions and body posture, translating them into natural English sentences.

## Features
- **Holistic Tracking**: Uses MediaPipe Holistic to capture 1662 landmarks (Face Mesh, Pose, and Both Hands).
- **Motion Recognition**: Employs an **LSTM (Long Short-Term Memory)** network to understand gestures over time.
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
   pip install opencv-python cvzone mediapipe tensorflow numpy scikit-learn
   ```

## Workflow

### 1. Data Collection
Collect data for each sign you want the AI to learn:
- **Hand-Only (Legacy)**: Run `python collect_sign_dataset.py` (static) or `python collect_landmarks.py` (motion).
- **Holistic Tracking (Advanced)**: Run `python collect_holistic.py`. Press **'s'** to record a 30-frame sequence including your face, pose, and hands.

### 2. Verification
Check your data quality before training:
- **Numerical View**: `python preview_npy.py <path_to_file.npy>`
- **Skeletal Animation**: `python visualize_landmarks.py <path_to_file.npy>`

### 3. Training
Train your custom LSTM model. The trainer dynamically detects if you are using Hand-Only (63 features) or Holistic (1662 features) data:
```bash
python train_model.py
```
This will generate `hand_model.h5` and `labels.txt` in the `Model/` folder.

### 4. Real-time Translation (CLI)
Run the desktop-based translation system to see the AI in action:
- **Hand-Only**: `python realtime_inference.py`
- **Holistic**: `python realtime_holistic.py`

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

**Web Dashboard Features:**
- **Live Translator**: Real-time sign-to-speech translation overlay with audio Text-To-Speech (TTS), reconstruction logs, and copy-to-clipboard.
- **Dataset Builder**: A dedicated capture panel with a live camera preview. You can record consecutive sequences under a single label without re-typing.
- **Model Trainer**: Configure training epochs/batches and watch real-time graphs and console output.
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

## Project Roadmap
See [ROADMAP.md](ROADMAP.md) for details:
- **Phase 1**: Landmark Collection (COMPLETED)
- **Phase 2**: Sequence Modeling (COMPLETED)
- **Phase 3**: Sentence Reconstruction (COMPLETED)
- **Phase 4**: Holistic Body/Face Tracking (COMPLETED)

## Project Structure
- `app.py`: Main entrypoint for the Interactive Web Dashboard.
- `collect_holistic.py`: Captures full-body holistic sequences (Phase 4).
- `realtime_holistic.py`: Live holistic translation script (Phase 4).
- `collect_sign_dataset.py`: Captures static hand landmarks (Legacy).
- `collect_landmarks.py`: Captures hand motion sequences (Legacy).
- `train_model.py`: Trains the LSTM network (Supports both Hand and Holistic).
- `realtime_inference.py`: Live hand-only translation (Legacy).
- `utils/`:
    - `holistic_utils.py`: Logic for extracting 1662 holistic data points.
    - `normalization.py`: Hand-only translation and scaling logic.
    - `translator.py`: Gloss-to-Text sentence reconstruction.
    - `logger_config.py`: Advanced logging configuration.
- `Data/`: Folder containing `.npy` datasets.
- `Model/`: Folder containing trained models and labels.

## Acknowledgments
- [cvzone](https://github.com/cvzone/cvzone) for simplified tracking utilities.
- [MediaPipe](https://mediapipe.dev/) for state-of-the-art coordinate extraction.
