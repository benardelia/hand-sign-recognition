import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
from tensorflow.keras.models import load_model
from utils.logger_config import setup_logger
from utils.normalization import normalize_landmarks

# Initialize Logger
logger = setup_logger(__name__)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join('Model', 'hand_model.h5')
LABELS_PATH = os.path.join('Model', 'labels.txt')
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21
NUM_COORDS = 3

# Load Model
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model not found at {MODEL_PATH}. Please run train_model.py first.")
    exit()

model = load_model(MODEL_PATH)
logger.info("Model loaded successfully.")

# Load Labels
with open(LABELS_PATH, 'r') as f:
    labels = f.read().splitlines()
logger.info(f"Loaded labels: {labels}")

# Initialize Webcam and Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Inference State
sequence = []
sentence = []
predictions = []
threshold = 0.8 # Confidence threshold

logger.info("--- Starting Real-time Inference ---")

while True:
    success, img = cap.read()
    if not success:
        break

    img_output = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        # Normalize landmarks (translation and scaling)
        landmarks = normalize_landmarks(lmList)
        
        # Add to sequence buffer
        sequence.append(landmarks)
        sequence = sequence[-SEQUENCE_LENGTH:] # Keep only the last 30 frames
        
        # When we have a full sequence, predict
        if len(sequence) == SEQUENCE_LENGTH:
            # Reshape for model: (1, 30, 63)
            input_data = np.expand_dims(sequence, axis=0)
            res = model.predict(input_data, verbose=0)[0]
            
            # Get the best prediction
            index = np.argmax(res)
            confidence = res[index]
            
            # Logic to build a sentence and avoid flickering
            if confidence > threshold:
                label = labels[index]
                
                # Draw the prediction on screen
                x, y, w, h = hand['bbox']
                cv2.rectangle(img_output, (x - 20, y - 60), (x + 120, y - 10), (0, 255, 0), cv2.FILLED)
                cv2.putText(img_output, f"{label} ({int(confidence*100)}%)", (x - 15, y - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # Clear sequence when no hand is detected to avoid predicting old data
        sequence = []

    # Display Instructions
    cv2.putText(img_output, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow("Hand Sign Translation", img_output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
logger.info("Inference ended.")
