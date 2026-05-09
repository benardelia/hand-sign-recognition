import cv2
import os
import numpy as np
import time
import mediapipe as mp
from utils.logger_config import setup_logger
from utils.holistic_utils import extract_holistic_keypoints

# Initialize Logger
logger = setup_logger(__name__)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
DATA_PATH = os.path.join('Data', 'Holistic_Landmarks')
SEQUENCE_LENGTH = 30
label = "Hello" # Change this label for each sign you want to collect
counter = 0

# Ensure directories exist
target_dir = os.path.join(DATA_PATH, label)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

cap = cv2.VideoCapture(0)

logger.info(f"--- Holistic Data Collection Mode: Ready to collect '{label}' ---")
logger.info("Press 'S' to start capturing a 30-frame sequence.")
logger.info("Press 'Q' to quit.")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.error("Failed to capture image from camera.")
            break

        # Recolor feed to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = holistic.process(image)

        # Recolor back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
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

        # Display info
        cv2.putText(image, f"Label: {label} | Collected: {counter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "[S] Start Sequence | [Q] Quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Holistic Dataset Collection", image)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
            
        elif key & 0xFF == ord('s'):
            sequence = []
            logger.info(f"Recording sequence {counter + 1}...")
            
            # Show visual cue that recording started
            cv2.putText(image, "RECORDING...", (int(image.shape[1]/2)-100, int(image.shape[0]/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow("Holistic Dataset Collection", image)
            cv2.waitKey(1000) # Wait 1 second before starting to let user prepare
            
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                
                # Recolor feed to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = holistic.process(image)

                # Extract holistic keypoints (1662 values)
                keypoints = extract_holistic_keypoints(results)
                sequence.append(keypoints)

                # Recolor back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks to show we are recording
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                cv2.putText(image, f"Recording Frame: {frame_num+1}/{SEQUENCE_LENGTH}", (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Holistic Dataset Collection", image)
                cv2.waitKey(10)
                
            # Save the sequence
            timestamp = int(time.time() * 1000)
            file_path = os.path.join(target_dir, f"holistic_seq_{timestamp}.npy")
            np.save(file_path, np.array(sequence))
            logger.info(f"Saved: {file_path}")
            
            counter += 1

cap.release()
cv2.destroyAllWindows()
logger.info("Session Ended.")
