# Hand Sign Recognition

A real-time hand sign detection and translation system using Python, OpenCV, and MediaPipe.

## Project Overview
This project is a high-performance **Landmark-based** recognition system. By extracting hand joint coordinates instead of raw images, the system is lightweight, fast, and capable of translating both static signs (letters) and dynamic gestures into full, natural English sentences using an LSTM neural network.

## Features
- **Landmark Extraction**: Uses MediaPipe to capture the (x, y, z) coordinates of 21 hand joints.
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
- **Static Signs (Letters)**: Run `python sign_dataset_capture.py` and press **'s'** to save a single frame.
- **Dynamic Phrases (Motion)**: Run `python collect_landmarks.py` and press **'s'** to record a 30-frame sequence.

### 2. Verification
Check your data quality before training:
- **Numerical View**: `python preview_npy.py <path_to_file.npy>`
- **Skeletal Animation**: `python visualize_landmarks.py <path_to_file.npy>`

### 3. Training
Train your custom LSTM model. The trainer automatically applies **Landmark Normalization** and **Data Augmentation** to improve accuracy:
```bash
python train_model.py
```
This will generate `hand_model.h5` and `labels.txt` in the `Model/` folder.

### 4. Real-time Translation
Run the translation system to see the AI in action:
```bash
python realtime_inference.py
```
- **Interact**: Press **'C'** to clear the current sentence and start fresh.
- **Quit**: Press **'Q'** to exit the application.

## Project Roadmap
See [ROADMAP.md](ROADMAP.md) for the detailed path:
- **Phase 1**: Landmark Collection (COMPLETED)
- **Phase 2**: Sequence Modeling (COMPLETED)
- **Phase 3**: Sentence Reconstruction (COMPLETED)
- **Phase 4**: Holistic Body/Face Tracking (ACTIVE)

## Project Structure
- `sign_dataset_capture.py`: Captures static landmarks.
- `collect_landmarks.py`: Captures motion sequences.
- `train_model.py`: Trains the LSTM neural network.
- `realtime_inference.py`: Live translation script with sentence reconstruction.
- `visualize_landmarks.py`: Animates and plays back captured data.
- `preview_npy.py`: Shows numerical data info.
- `utils/`:
    - `logger_config.py`: Advanced logging configuration.
    - `normalization.py`: Landmark translation and scaling logic.
    - `translator.py`: Gloss-to-Text sentence reconstruction.
- `Data/`: Folder containing the `.npy` datasets.
- `Model/`: Folder containing the trained model and labels.

## Acknowledgments
- [cvzone](https://github.com/cvzone/cvzone) for simplified hand tracking.
- [MediaPipe](https://mediapipe.dev/) for coordinate extraction.
