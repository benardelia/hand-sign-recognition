# Hand Sign Recognition

A real-time hand sign detection and translation system using Python, OpenCV, and MediaPipe.

## Project Overview
This project is evolving from static image classification to a high-performance **Landmark-based** recognition system. By capturing hand joint coordinates instead of raw images, the system is lighter, faster, and ready for advanced sequence modeling (LSTM).

## Features
- **Landmark Extraction**: Uses MediaPipe to capture the (x, y, z) coordinates of 21 hand joints.
- **Advanced Logging**: Color-coded console logs with clickable file links for easier debugging.
- **Static & Sequence Collection**: Support for both single-post capture and motion-sequence recording.
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
   pip install opencv-python cvzone mediapipe tensorflow numpy
   ```

## Usage

### 1. Collect Static Signs (Letters)
Use `sign_dataset_capture.py` to collect landmark data for static hand poses.
- Run: `python sign_dataset_capture.py`
- Press **'s'** to save the current hand landmarks to `Data/C/` as a `.npy` file.

### 2. Collect Dynamic Sequences (Phrases)
Use `collect_landmarks.py` to record motion sequences (Phase 1 of the Roadmap).
- Run: `python collect_landmarks.py`
- Press **'s'** to record a **30-frame sequence** (approx. 1 second of motion).
- Files are saved in `Data/Landmarks/`.

### 3. Verify Your Data
Before training, use the verification scripts to check your data quality:

- **Numerical Preview**:
  ```bash
  python preview_npy.py <path_to_file.npy>
  ```
- **Visual Animation**:
  ```bash
  python visualize_landmarks.py <path_to_file.npy>
  ```

## Project Roadmap
See [ROADMAP.md](ROADMAP.md) for the detailed path toward full sentence translation, including:
- **Phase 1**: Landmark Collection (Status: Active)
- **Phase 2**: Sequence Modeling (LSTM)
- **Phase 3**: NLP Sentence Reconstruction
- **Phase 4**: Holistic Body/Face Tracking

## Project Structure
- `sign_dataset_capture.py`: Captures static landmarks.
- `collect_landmarks.py`: Captures motion sequences.
- `visualize_landmarks.py`: Animates and plays back captured data.
- `preview_npy.py`: Shows numerical data info.
- `utils/logger_config.py`: Advanced logging configuration.
- `Data/`: Folder containing the `.npy` datasets.

## Acknowledgments
- [cvzone](https://github.com/cvzone/cvzone) for simplified hand tracking.
- [MediaPipe](https://mediapipe.dev/) for coordinate extraction.
