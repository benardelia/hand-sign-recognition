# Hand Sign Recognition

A real-time hand sign detection and classification system using Python, OpenCV, and MediaPipe.

## Project Overview
This project provides a complete pipeline for building a hand sign dataset and using it for real-time classification. It uses `cvzone`'s Hand Tracking and Classification modules, built on top of MediaPipe and TensorFlow.

## Features
- **Data Collection**: Capture hand sign images with automatic cropping and resizing to a uniform 300x300 format.
- **Hand Tracking**: Real-time multi-hand tracking with skeletal visualization.
- **Classification**: Real-time sign classification using a trained deep learning model.

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
   pip install opencv-python cvzone mediapipe tensorflow
   ```

## Usage

### 1. Capture Dataset
Use `sign_dataset_capture.py` to collect images for each hand sign class.
- Run the script: `python sign_dataset_capture.py`
- Position your hand in the camera view.
- Press **'s'** to save the processed crop of your hand to the `Data/C` folder.
- Press **'q'** to exit.

> [!NOTE] 
> You can change the target folder for different signs by modifying the `folder` variable in `sign_dataset_capture.py`.

### 2. Train Your Model
After collecting enough data (e.g., 300+ images per class):
1. Go to [Teachable Machine](https://teachablemachine.withgoogle.com/train/image).
2. Upload your captured images for each class.
3. Train the model and export it as a **TensorFlow / Keras** model.
4. Place `keras_model.h5` and `labels.txt` inside a folder named `Model/`.

### 3. Run Recognition
Use `sign_detector.py` to detect signs in real-time.
- Run the script: `python sign_detector.py`
- The script will display the predicted label above your hand.
- Press **'q'** to exit.

## Project Structure
- `sign_dataset_capture.py`: Script for collecting training data.
- `sign_detector.py`: Script for real-time recognition.
- `Data/`: Folder containing the captured image datasets.
- `Model/`: Folder containing the trained `.h5` model and `labels.txt`.

## Acknowledgments
- [cvzone](https://github.com/cvzone/cvzone) for the simplified hand tracking wrappers.
- [MediaPipe](https://mediapipe.dev/) for the underlying hand tracking technology.
