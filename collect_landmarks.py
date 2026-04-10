import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import time
from utils.logger_config import setup_logger

# Initialize Logger
logger = setup_logger(__name__)

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Configuration
folder = "Data/Landmarks"
label = "A"  # Change this for different signs
sequence_length = 30  # Number of frames per gesture
sequences_to_collect = 30  # How many sequences to collect for this label

# Ensure directory exists
target_dir = os.path.join(folder, label)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

logger.info(f"Starting landmark collection for label: {label}")
logger.info("Instructions:")
logger.info("1. Position your hand in the camera view.")
logger.info("2. Press 's' to start recording a 30-frame sequence.")
logger.info("3. Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hand
    hands, img = detector.findHands(img)
    
    cv2.imshow("Landmark Collection", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('s'):
        logger.info(f"Recording sequence...")
        sequence = []
        
        # Collect frames for the sequence
        count = 0
        while count < sequence_length:
            success, img = cap.read()
            if not success:
                break
                
            hands, img = detector.findHands(img)
            
            if hands:
                hand = hands[0]
                lmList = hand['lmList'] # 21 landmarks [x, y, z]
                
                # Flatten the list of landmarks into a single 1D array (21*3 = 63 values)
                landmarks = np.array(lmList).flatten()
                sequence.append(landmarks)
                count += 1
            
            cv2.imshow("Landmark Collection", img)
            cv2.waitKey(1)
        
        # Save the sequence
        if len(sequence) == sequence_length:
            timestamp = int(time.time() * 1000)
            file_path = os.path.join(target_dir, f"seq_{timestamp}.npy")
            np.save(file_path, np.array(sequence))
            logger.info(f"Sequence saved to {file_path}")
        else:
            logger.warning("Failed to collect a complete sequence. Hand lost during recording.")

cap.release()
cv2.destroyAllWindows()
logger.info("Collection finished.")
