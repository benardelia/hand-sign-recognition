import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from utils.logger_config import setup_logger
from utils.normalization import normalize_landmarks
from utils.translator import translator

# Initialize Logger
logger = setup_logger(__name__)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join('Model', 'hand_model.h5')
LABELS_PATH = os.path.join('Model', 'labels.txt')
SEQUENCE_LENGTH = 30
STABILITY_FRAMES = 5 # Number of consecutive frames a word must be detected

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

# Inference & Sentence State
sequence = []
gloss_buffer = []  # To store the sequence of detected keywords
current_sentence = "Start signing..."
last_prediction = ""
action_counter = 0

threshold = 0.85 # High confidence threshold for stability

logger.info("--- Starting Real-time Phase 3 Inference ---")

while True:
    success, img = cap.read()
    if not success:
        break

    img_output = img.copy()
    
    # 1. Detect hands and landmarks
    # draw=False keeps the UI cleaner
    hands, img = detector.findHands(img, draw=False)
    
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        
        # 2. Normalize and update sequence buffer
        landmarks = normalize_landmarks(lmList)
        sequence.append(landmarks)
        sequence = sequence[-SEQUENCE_LENGTH:]
        
        # 3. Predict when buffer is full
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            res = model.predict(input_data, verbose=0)[0]
            index = np.argmax(res)
            confidence = res[index]
            
            # 4. Debouncing & Stability Logic
            if confidence > threshold:
                detected_word = labels[index]
                
                if detected_word == last_prediction:
                    action_counter += 1
                else:
                    action_counter = 0
                    last_prediction = detected_word
                
                # If word is stable for N frames, add it to sentence
                if action_counter == STABILITY_FRAMES:
                    # Check if it's a new word to avoid repeat additions
                    if not gloss_buffer or gloss_buffer[-1] != detected_word:
                        gloss_buffer.append(detected_word)
                        # Remove noise (keep only last 5 words for short sentence context)
                        gloss_buffer = gloss_buffer[-5:] 
                        # Translate glosses to natural sentence
                        current_sentence = translator.translate(gloss_buffer)
                        logger.info(f"Detected: {detected_word} | Sentence: {current_sentence}")
                    
                    action_counter = 0 # Reset to allow re-detecting if significant movement occurs
                
                # Visual Indicator for current prediction
                x, y, w, h = hand['bbox']
                color = (0, 255, 0) if action_counter >= STABILITY_FRAMES-1 else (0, 255, 255)
                cv2.rectangle(img_output, (x - 20, y - 60), (x + 150, y - 10), color, cv2.FILLED)
                cv2.putText(img_output, f"{detected_word} {int(confidence*100)}%", (x - 15, y - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        sequence = []
        action_counter = 0

    # 5. UI Overlay
    # Draw Background rectangle for sentence
    cv2.rectangle(img_output, (0, img.shape[0] - 80), (img.shape[1], img.shape[0]), (50, 50, 50), cv2.FILLED)
    
    # Draw current translated sentence
    cv2.putText(img_output, current_sentence, (20, img.shape[0] - 30), 
                cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

    # Instructions
    cv2.putText(img_output, "[Q] Quit  [C] Clear Sentence", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Sign-to-Sentence Translator", img_output)
    
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
