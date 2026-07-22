import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
if hasattr(cv2, 'setLogLevel'):
    cv2.setLogLevel(0)

import time
import json
import sys
import numpy as np
import threading
import subprocess
import mediapipe as mp
try:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except Exception:
    try:
        from mediapipe import solutions as mp_solutions
        mp_holistic = mp_solutions.holistic
        mp_drawing = mp_solutions.drawing_utils
        mp_face_mesh = mp_solutions.face_mesh
    except Exception:
        mp_holistic = getattr(mp, 'solutions').holistic
        mp_drawing = getattr(mp, 'solutions').drawing_utils
        mp_face_mesh = getattr(mp, 'solutions').face_mesh

import base64
from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model

from utils.logger_config import setup_logger
from utils.holistic_utils import extract_holistic_keypoints
from utils.translator import translator

# Initialize Logger
logger = setup_logger(__name__)

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join('Model', 'hand_model.h5')
LABELS_PATH = os.path.join('Model', 'labels.txt')
DATA_PATH = os.path.join('Data', 'Holistic_Landmarks')
SEQUENCE_LENGTH = 30
STABILITY_FRAMES = 5
PORT = 5001

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Frame buffers
        self.frame = None
        self.annotated_frame = None
        
        # ML Model and labels
        self.model = None
        self.labels = []
        self.expected_shape = 1662
        self.model_loaded = False
        self.load_ml_model()
        
        # Prediction states
        self.sequence = []
        self.gloss_buffer = []
        self.current_sentence = "Start signing..."
        self.last_prediction = ""
        self.confidence = 0.0
        self.action_counter = 0
        self.threshold = 0.85
        
        # Recording states
        self.recording_label = None
        self.recording_status = "idle"  # "idle", "countdown", "recording", "saved"
        self.recording_frames_collected = 0
        self.recording_sequence = []
        self.countdown_start_time = 0
        self.countdown_duration = 3.0 # seconds
        
        # Camera selection state
        self.camera_index = 0
        self.camera_needs_switch = False
        self.pending_camera_index = 0
        
    def load_ml_model(self):
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
                self.model = load_model(MODEL_PATH)
                self.expected_shape = self.model.input_shape[-1]
                with open(LABELS_PATH, 'r') as f:
                    self.labels = f.read().splitlines()
                self.model_loaded = True
                logger.info(f"Model loaded successfully. Labels: {self.labels}. Expected feature size: {self.expected_shape}")
            else:
                self.model_loaded = False
                logger.info("No trained model or labels found in Model/ folder. Real-time translation is paused until a model is trained via the Web Dashboard.")
        except Exception as e:
            self.model_loaded = False
            logger.error(f"Failed to load model: {e}")
            
    def reload_model(self):
        with self.lock:
            self.load_ml_model()
            self.sequence = []
            self.gloss_buffer = []
            self.current_sentence = "Model reloaded. Start signing..."
            self.last_prediction = ""
            self.confidence = 0.0
            
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("CameraManager thread started.")
            
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info("CameraManager thread stopped.")
        
    def _run(self):
        # Initialize camera in the thread
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.warning("No hardware webcam detected on server (Headless/VPS environment). Local camera capture is disabled.")
            self.is_running = False
            return
            
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.is_running:
                # Check for camera switch request
                if self.camera_needs_switch:
                    logger.info(f"CameraManager thread switching to camera index: {self.pending_camera_index}")
                    if self.cap and self.cap.isOpened():
                        self.cap.release()
                    self.camera_index = self.pending_camera_index
                    self.cap = cv2.VideoCapture(self.camera_index)
                    self.camera_needs_switch = False
                    if not self.cap.isOpened():
                        logger.error(f"Failed to open camera index {self.camera_index}. Reverting to 0.")
                        self.camera_index = 0
                        self.cap = cv2.VideoCapture(0)
                        
                success, frame = self.cap.read()
                if not success:
                    time.sleep(0.03)
                    continue
                
                self.process_frame_data(frame, holistic, mp_drawing, mp_holistic)
                time.sleep(0.01)

    def process_frame_data(self, raw_frame, holistic_instance=None, drawing_utils=None, holistic_module=None):
        if drawing_utils is None:
            drawing_utils = mp_drawing
        if holistic_module is None:
            holistic_module = mp_holistic
            
        frame = cv2.flip(raw_frame, 1)
        self.frame = frame.copy()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        if holistic_instance is not None:
            results = holistic_instance.process(image)
        else:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                results = holistic.process(image)
                
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.face_landmarks:
            drawing_utils.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                     drawing_utils.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     drawing_utils.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        if results.pose_landmarks:
            drawing_utils.draw_landmarks(image, results.pose_landmarks, holistic_module.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            drawing_utils.draw_landmarks(image, results.left_hand_landmarks, holistic_module.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            drawing_utils.draw_landmarks(image, results.right_hand_landmarks, holistic_module.HAND_CONNECTIONS)
        
        keypoints = extract_holistic_keypoints(results)
        
        with self.lock:
            if self.recording_status == "countdown":
                elapsed = time.time() - self.countdown_start_time
                remaining = max(0.0, self.countdown_duration - elapsed)
                
                h, w, _ = image.shape
                cv2.rectangle(image, (0, 0), (w, h), (0, 0, 0), 2)
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
                
                text = f"{int(remaining) + 1}" if remaining > 0 else "GO!"
                cv2.putText(image, text, (int(w/2) - 30, int(h/2) + 20), 
                            cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 255), 5, cv2.LINE_AA)
                cv2.putText(image, f"Prepare sign: '{self.recording_label}'", (20, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                if elapsed >= self.countdown_duration:
                    self.recording_status = "recording"
                    self.recording_frames_collected = 0
                    self.recording_sequence = []
                    logger.info("Recording started...")
                    
            elif self.recording_status == "recording":
                self.recording_sequence.append(keypoints)
                self.recording_frames_collected += 1
                
                h, w, _ = image.shape
                cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 3)
                cv2.putText(image, "RECORDING...", (15, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Frame: {self.recording_frames_collected}/{SEQUENCE_LENGTH}", (15, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                progress_w = int((self.recording_frames_collected / SEQUENCE_LENGTH) * (w - 40))
                cv2.rectangle(image, (20, h - 30), (w - 20, h - 15), (50, 50, 50), -1)
                cv2.rectangle(image, (20, h - 30), (20 + progress_w, h - 15), (0, 0, 255), -1)
                
                if self.recording_frames_collected >= SEQUENCE_LENGTH:
                    self.recording_status = "saving"
                    threading.Thread(target=self._save_recording, daemon=True).start()
                    
            elif self.recording_status == "saved":
                h, w, _ = image.shape
                cv2.rectangle(image, (0, 0), (w, h), (0, 255, 0), 3)
                cv2.putText(image, "SEQUENCE SAVED!", (int(w/2) - 150, int(h/2)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
            elif self.model_loaded:
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-SEQUENCE_LENGTH:]
                
                if len(self.sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(self.sequence, axis=0)
                    
                    if self.expected_shape == 1662:
                        res = self.model.predict(input_data, verbose=0)[0]
                        index = np.argmax(res)
                        confidence = res[index]
                        self.confidence = float(confidence)
                        
                        if confidence > self.threshold:
                            detected_word = self.labels[index]
                            
                            if detected_word == self.last_prediction:
                                self.action_counter += 1
                            else:
                                self.action_counter = 0
                                self.last_prediction = detected_word
                            
                            if self.action_counter == STABILITY_FRAMES:
                                if not self.gloss_buffer or self.gloss_buffer[-1] != detected_word:
                                    self.gloss_buffer.append(detected_word)
                                    self.gloss_buffer = self.gloss_buffer[-5:]
                                    self.current_sentence = translator.translate(self.gloss_buffer)
                                    logger.info(f"Detected gloss: {detected_word} | Reconstructed sentence: {self.current_sentence}")
                                self.action_counter = 0
                                
                            cv2.putText(image, f"{detected_word} ({int(confidence*100)}%)", (15, 45), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "Error: Model shape mismatch.", (15, 45), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        self.annotated_frame = image.copy()
        return image

    def _save_recording(self):
        try:
            target_dir = os.path.join(DATA_PATH, self.recording_label)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            timestamp = int(time.time() * 1000)
            file_path = os.path.join(target_dir, f"holistic_seq_{timestamp}.npy")
            np.save(file_path, np.array(self.recording_sequence))
            logger.info(f"Successfully saved landmark sequence: {file_path}")
            
            with self.lock:
                self.recording_status = "saved"
            
            # Keep "saved" display for 1.5 seconds, then return to idle
            time.sleep(1.5)
            
            with self.lock:
                self.recording_status = "idle"
                self.recording_label = None
                self.recording_sequence = []
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            with self.lock:
                self.recording_status = "idle"
                self.recording_label = None

    def start_recording(self, label):
        with self.lock:
            if self.recording_status == "idle":
                self.recording_label = label
                self.recording_status = "countdown"
                self.countdown_start_time = time.time()
                logger.info(f"Initiated countdown for recording label: {label}")
                return True
            return False

    def clear_sentence(self):
        with self.lock:
            self.gloss_buffer = []
            self.current_sentence = "Sentence Cleared."
            self.last_prediction = ""
            self.confidence = 0.0
            logger.info("Sentence buffer cleared.")

    def get_jpeg_frame(self):
        with self.lock:
            if self.annotated_frame is None:
                # Return informative placeholder frame if camera is offline/headless
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                msg = "No Server Webcam (Client Stream Active)" if not self.is_running else "Starting Camera..."
                cv2.putText(img, msg, (80, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', img)
                return jpeg.tobytes()
                
            _, jpeg = cv2.imencode('.jpg', self.annotated_frame)
            return jpeg.tobytes()

# Global camera manager instance
camera_manager = CameraManager()

# Global variables for training logs streaming
training_process = None
training_logs = []

def get_sign_directories():
    if not os.path.exists(DATA_PATH):
        return []
    dirs = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    return sorted(dirs)

# --- SERVER ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_jpeg_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.03) # ~30 FPS

@app.route('/video_feed')
def video_feed():
    camera_manager.start()
    return Response(gen(camera_manager),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# MediaPipe instance lazily initialized for HTTP client stream processing
api_holistic_instance = None
api_holistic_lock = threading.Lock()

def get_api_holistic():
    global api_holistic_instance
    if api_holistic_instance is None:
        with api_holistic_lock:
            if api_holistic_instance is None:
                api_holistic_instance = mp_holistic.Holistic(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
                )
    return api_holistic_instance

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json or {}
        image_data = data.get('image', '')
        if not image_data:
            return jsonify({"status": "error", "message": "No image payload provided"}), 400
            
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"status": "error", "message": "Invalid image payload"}), 400
            
        holistic_inst = get_api_holistic()
        annotated_frame = camera_manager.process_frame_data(
            frame, 
            holistic_inst, 
            mp_drawing, 
            mp_holistic
        )
        
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        encoded_image = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        with camera_manager.lock:
            return jsonify({
                "status": "success",
                "annotated_image": encoded_image,
                "current_sentence": camera_manager.current_sentence,
                "last_prediction": camera_manager.last_prediction,
                "confidence": round(camera_manager.confidence, 2),
                "gloss_buffer": camera_manager.gloss_buffer,
                "is_recording": camera_manager.recording_status != "idle",
                "recording_label": camera_manager.recording_label or "",
                "recording_status": camera_manager.recording_status,
                "recording_frames_collected": camera_manager.recording_frames_collected,
                "model_loaded": camera_manager.model_loaded,
                "labels": camera_manager.labels
            })
    except Exception as e:
        logger.error(f"Error processing client frame: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status')
def get_status():
    with camera_manager.lock:
        return jsonify({
            "current_sentence": camera_manager.current_sentence,
            "last_prediction": camera_manager.last_prediction,
            "confidence": round(camera_manager.confidence, 2),
            "gloss_buffer": camera_manager.gloss_buffer,
            "is_recording": camera_manager.recording_status != "idle",
            "recording_label": camera_manager.recording_label,
            "recording_status": camera_manager.recording_status,
            "recording_frames_collected": camera_manager.recording_frames_collected,
            "model_loaded": camera_manager.model_loaded,
            "labels": camera_manager.labels
        })

@app.route('/api/clear_sentence', methods=['POST'])
def clear_sentence():
    camera_manager.clear_sentence()
    return jsonify({"status": "success"})

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    data = request.json or {}
    label = data.get('label', '').strip()
    if not label:
        return jsonify({"status": "error", "message": "Label name is required."}), 400
        
    success = camera_manager.start_recording(label)
    if success:
        return jsonify({"status": "success", "message": f"Countdown started for '{label}'"})
    else:
        return jsonify({"status": "error", "message": "Camera is busy recording."}), 409

@app.route('/api/signs')
def get_signs():
    signs = []
    dirs = get_sign_directories()
    for d in dirs:
        p = os.path.join(DATA_PATH, d)
        count = len([f for f in os.listdir(p) if f.endswith('.npy')])
        signs.append({"name": d, "samples": count})
    return jsonify({"signs": signs})

@app.route('/api/npy_files')
def get_npy_files():
    result = {}
    dirs = get_sign_directories()
    for d in dirs:
        p = os.path.join(DATA_PATH, d)
        files = sorted([f for f in os.listdir(p) if f.endswith('.npy')])
        result[d] = [{"filename": f, "path": os.path.join(p, f)} for f in files]
    return jsonify(result)

@app.route('/api/npy_data')
def get_npy_data():
    filepath = request.args.get('path', '')
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
        
    try:
        data = np.load(filepath)
        # Check if data shape is sequence (30, 1662)
        if data.ndim == 2 and data.shape[0] == 30 and data.shape[1] == 1662:
            frames_list = []
            for frame_idx in range(30):
                frame_data = data[frame_idx]
                
                # Split coordinates according to extract_holistic_keypoints:
                # Pose: 33 landmarks * 4 coords = 132 elements
                # Face: 468 landmarks * 3 coords = 1404 elements
                # Left Hand: 21 landmarks * 3 coords = 63 elements
                # Right Hand: 21 landmarks * 3 coords = 63 elements
                
                pose_part = frame_data[0:132].reshape(33, 4)
                face_part = frame_data[132:1536].reshape(468, 3)
                lh_part = frame_data[1536:1599].reshape(21, 3)
                rh_part = frame_data[1599:1662].reshape(21, 3)
                
                # Format landmarks into JSON-friendly dicts
                frames_list.append({
                    "pose": [{"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2]), "visibility": float(lm[3])} for lm in pose_part],
                    "face": [{"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2])} for lm in face_part],
                    "left_hand": [{"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2])} for lm in lh_part],
                    "right_hand": [{"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2])} for lm in rh_part]
                })
            return jsonify({
                "filename": os.path.basename(filepath),
                "type": "holistic",
                "frames": frames_list
            })
        else:
            return jsonify({"error": f"Invalid data format. Expected shape (30, 1662), got shape {data.shape}"}), 400
    except Exception as e:
        logger.error(f"Error loading .npy file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cameras')
def get_cameras():
    active_idx = getattr(camera_manager, 'camera_index', 0)
    available = []
    
    # We probe indices 0 to 4 safely
    for index in range(5):
        if index == active_idx and camera_manager.cap and camera_manager.cap.isOpened():
            name = f"Camera {index} (Active)"
            available.append({"id": index, "name": name, "active": True})
            continue
            
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                name = f"Camera {index}"
                if index == 0:
                    name += " (Default)"
                available.append({"id": index, "name": name, "active": False})
                cap.release()
        except Exception:
            pass
            
    if not available:
        available.append({"id": 0, "name": "Webcam (Client Browser Stream)", "active": True})
        
    available.sort(key=lambda x: x["id"])
    return jsonify({"cameras": available, "active_id": active_idx})

@app.route('/api/select_camera', methods=['POST'])
def select_camera():
    data = request.json or {}
    camera_id = data.get('camera_id')
    if camera_id is None:
        return jsonify({"status": "error", "message": "Camera ID is required."}), 400
        
    try:
        camera_id = int(camera_id)
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid Camera ID."}), 400
        
    # Request switch
    camera_manager.pending_camera_index = camera_id
    camera_manager.camera_needs_switch = True
    
    # Wait a brief moment for the thread to process the switch
    time.sleep(0.6)
    
    # Check if the active index changed successfully
    if camera_manager.camera_index == camera_id:
        return jsonify({"status": "success", "message": f"Successfully switched to Camera {camera_id}."})
    else:
        return jsonify({"status": "error", "message": f"Failed to switch to Camera {camera_id}."}), 500

@app.route('/api/delete_file', methods=['POST'])
def delete_file():
    import shutil
    data = request.json or {}
    filepath = data.get('path', '')
    if not filepath:
        return jsonify({"status": "error", "message": "File path is required."}), 400
        
    abs_data_path = os.path.abspath(DATA_PATH)
    abs_file_path = os.path.abspath(filepath)
    if not abs_file_path.startswith(abs_data_path):
        return jsonify({"status": "error", "message": "Unauthorized path access."}), 403
        
    if not os.path.exists(abs_file_path):
        return jsonify({"status": "error", "message": "File not found."}), 404
        
    try:
        os.remove(abs_file_path)
        logger.info(f"Successfully deleted file: {abs_file_path}")
        return jsonify({"status": "success", "message": "File deleted successfully."})
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/delete_sign', methods=['POST'])
def delete_sign():
    import shutil
    data = request.json or {}
    label = data.get('label', '').strip()
    if not label:
        return jsonify({"status": "error", "message": "Sign label is required."}), 400
        
    target_dir = os.path.abspath(os.path.join(DATA_PATH, label))
    abs_data_path = os.path.abspath(DATA_PATH)
    if not target_dir.startswith(abs_data_path) or target_dir == abs_data_path:
        return jsonify({"status": "error", "message": "Unauthorized path access."}), 403
        
    if not os.path.exists(target_dir):
        return jsonify({"status": "error", "message": "Sign dataset directory not found."}), 404
        
    try:
        shutil.rmtree(target_dir)
        logger.info(f"Successfully deleted sign dataset folder: {target_dir}")
        return jsonify({"status": "success", "message": "Sign dataset deleted successfully."})
    except Exception as e:
        logger.error(f"Failed to delete sign dataset folder: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def run_train():
    global training_process, training_logs
    if training_process and training_process.poll() is None:
        return jsonify({"status": "error", "message": "Training is already in progress."}), 409
        
    training_logs = []
    
    # We pause camera inference while training to free system resources
    camera_manager.stop()
    
    try:
        # Run train_model.py in background, capture stdout
        cmd = [sys.executable, "train_model.py"]
        # Set environment variable to force stdout flush
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        logger.info("Training process initiated.")
        return jsonify({"status": "success", "message": "Training started."})
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stream_train_logs')
def stream_train_logs():
    def generate_logs():
        global training_process, training_logs
        if not training_process:
            yield "data: {\"type\": \"log\", \"message\": \"No training process active.\"}\n\n"
            return
            
        logger.info("SSE client connected to training log stream.")
        
        while True:
            line = training_process.stdout.readline()
            if not line:
                if training_process.poll() is not None:
                    # Process completed
                    rc = training_process.returncode
                    msg = "Training complete!" if rc == 0 else f"Training failed with exit code {rc}"
                    logger.info(msg)
                    
                    # Reload the newly trained model in the CameraManager
                    if rc == 0:
                        camera_manager.reload_model()
                        
                    yield f"data: {json.dumps({'type': 'status', 'status': 'completed', 'code': rc, 'message': msg})}\n\n"
                    break
                time.sleep(0.1)
                continue
                
            stripped_line = line.strip()
            # Check if line contains Keras progress or special metrics
            # E.g. Epoch 1/150 or our custom printed callback EPOCH_END
            if "EPOCH_END:" in stripped_line:
                # Format: EPOCH_END:epoch:accuracy:loss:val_accuracy:val_loss
                parts = stripped_line.split(":")
                if len(parts) >= 6:
                    epoch = int(parts[1])
                    acc = float(parts[2])
                    loss = float(parts[3])
                    val_acc = float(parts[4])
                    val_loss = float(parts[5])
                    
                    data = {
                        "type": "epoch_metrics",
                        "epoch": epoch,
                        "accuracy": acc,
                        "loss": loss,
                        "val_accuracy": val_acc,
                        "val_loss": val_loss
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            else:
                data = {
                    "type": "log",
                    "message": stripped_line
                }
                yield f"data: {json.dumps(data)}\n\n"
                
    return Response(generate_logs(), mimetype='text/event-stream')

if __name__ == '__main__':
    import socket
    
    def find_free_port(start_port):
        port = start_port
        for _ in range(10):  # Try up to 10 consecutive ports
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) != 0:
                    return port
            port += 1
        return start_port

    target_port = int(os.environ.get('PORT', PORT))
    actual_port = find_free_port(target_port)
    
    if actual_port != target_port:
        logger.warning(f"Port {target_port} was in use. Falling back to port {actual_port}.")
    else:
        logger.info(f"Starting server on port {actual_port}...")

    # Start the CameraManager
    camera_manager.start()
    try:
        # Run Flask server
        app.run(host='0.0.0.0', port=actual_port, debug=False, threaded=True)
    finally:
        camera_manager.stop()
