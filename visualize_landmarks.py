import cv2
import numpy as np
import os
import sys
import time
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def draw_skeleton(img, landmarks):
    """Utility to draw custom hand skeleton connections."""
    
    # Detect if data is normalized (small values near 0)
    is_normalized = np.max(np.abs(landmarks)) < 10
    
    display_lms = landmarks.copy()
    if is_normalized:
        # Scale up and move to the center of the 1000x800 canvas
        scale = 300
        offset_x, offset_y = 500, 400
        display_lms[:, 0] = display_lms[:, 0] * scale + offset_x
        display_lms[:, 1] = display_lms[:, 1] * scale + offset_y
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8), # Index
        (5, 9), (9, 10), (10, 11), (11, 12), # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky & Palm
    ]
    
    # Draw lines
    for start, end in connections:
        p1 = (int(display_lms[start][0]), int(display_lms[start][1]))
        p2 = (int(display_lms[end][0]), int(display_lms[end][1]))
        cv2.line(img, p1, p2, (255, 255, 255), 2)
        
    # Draw joints
    for lm in display_lms:
        cv2.circle(img, (int(lm[0]), int(lm[1])), 6, (0, 255, 0), cv2.FILLED)

def visualize(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    try:
        data = np.load(file_path)
        logger.info(f"Visualizing: {os.path.basename(file_path)}")

        if data.ndim == 1:
            frames = [data.reshape(21, 3)]
            mode = "Static"
        else:
            frames = [f.reshape(21, 3) for f in data]
            mode = "Sequence"

        window_name = f"Visualization - {mode}"
        cv2.namedWindow(window_name)

        loops = 3 if mode == "Sequence" else 1
        for loop in range(loops):
            if mode == "Sequence":
                logger.info(f"Loop {loop + 1}/{loops}...")
            
            for i, landmarks in enumerate(frames):
                img = np.zeros((800, 1000, 3), np.uint8)
                draw_skeleton(img, landmarks)

                cv2.putText(img, f"Mode: {mode}", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f"File: {os.path.basename(file_path)}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                if mode == "Sequence":
                    cv2.putText(img, f"Frame: {i+1}/{len(frames)}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow(window_name, img)
                wait_time = 50 if mode == "Sequence" else 0
                key = cv2.waitKey(wait_time)
                if key & 0xFF == ord('q'): return

            if mode == "Sequence": time.sleep(0.5)

        if mode == "Static":
            logger.info("Static preview. Press any key in the window to close.")
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        logger.info("Visualization complete.")

    except Exception as e:
        logger.error(f"Error during visualization: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize(sys.argv[1])
    else:
        logger.warning("No file provided.")
        print("Usage: python visualize_landmarks.py <path_to_npy_file>")
