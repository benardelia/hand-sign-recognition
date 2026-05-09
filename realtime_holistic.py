import cv2
import numpy as np
import os
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils.logger_config import setup_logger
from utils.holistic_utils import extract_holistic_keypoints
from utils.translator import translator

# Initialize Logger
logger = setup_logger(__name__)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join('Model', 'hand_model.h5')
LABELS_PATH = os.path.join('Model', 'labels.txt')
SEQUENCE_LENGTH = 30
STABILITY_FRAMES = 5

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load Model
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model not found at {MODEL_PATH}. Please run train_model.py first.")
    exit()

model = load_model(MODEL_PATH)
logger.info("Model loaded successfully.")

# Determine if model expects holistic (1662) or hand-only (63) based on its input shape
expected_shape = model.input_shape[-1]
logger.info(f"Model expects input feature size: {expected_shape}")
if expected_shape != 1662:
    logger.warning("This script is designed for Holistic models (1662 features). Your model seems to be trained on hands only. Predictions may fail or be inaccurate.")

# Load Labels
with open(LABELS_PATH, 'r') as f:
    labels = f.read().splitlines()
logger.info(f"Loaded labels: {labels}")

cap = cv2.VideoCapture(0)

# Inference State
sequence = []
gloss_buffer = []
current_sentence = "Start signing..."
last_prediction = ""
action_counter = 0

threshold = 0.85 

logger.info("--- Starting Real-time Holistic Inference ---")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks for visual feedback
        # 1. Face
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        # 2. Pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # 3. Left Hand
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # 4. Right Hand
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract holistic keypoints
        keypoints = extract_holistic_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            # Predict
            input_data = np.expand_dims(sequence, axis=0)
            
            # Simple check to avoid crashing if model expects 63 but gets 1662
            if expected_shape == 1662:
                res = model.predict(input_data, verbose=0)[0]
                index = np.argmax(res)
                confidence = res[index]
                
                # Debouncing & Stability
                if confidence > threshold:
                    detected_word = labels[index]
                    
                    if detected_word == last_prediction:
                        action_counter += 1
                    else:
                        action_counter = 0
                        last_prediction = detected_word
                    
                    if action_counter == STABILITY_FRAMES:
                        if not gloss_buffer or gloss_buffer[-1] != detected_word:
                            gloss_buffer.append(detected_word)
                            gloss_buffer = gloss_buffer[-5:] 
                            current_sentence = translator.translate(gloss_buffer)
                            logger.info(f"Detected: {detected_word} | Sentence: {current_sentence}")
                        
                        action_counter = 0 
                    
                    # Display current prediction in top left
                    cv2.putText(image, f"{detected_word} {int(confidence*100)}%", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Model expects 63 features, got 1662.", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # UI Overlay
        cv2.rectangle(image, (0, image.shape[0] - 80), (image.shape[1], image.shape[0]), (50, 50, 50), cv2.FILLED)
        cv2.putText(image, current_sentence, (20, image.shape[0] - 30), 
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(image, "[Q] Quit  [C] Clear Sentence", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Holistic Sign Translator", image)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            gloss_buffer = []
            current_sentence = "Sentence Cleared."
            logger.info("Sentence buffer cleared.")

cap.release()
cv2.destroyAllWindows()
logger.info("Session Ended.")
