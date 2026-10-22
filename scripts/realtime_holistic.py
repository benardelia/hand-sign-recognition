"""
Real-time holistic sign translation from a local webcam (desktop / experimental track).

    python scripts/realtime_holistic.py --profile experimental
    python scripts/realtime_holistic.py --profile low --camera 1

Camera resolution and sequence length come from the profile (config/profiles.json),
the model from Model/<profile>/ (see ml/train_model.py).
"""
import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

# Run from the repo root (Data/, Model/, config/ are relative to it); make utils/ importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.features import get_profile
from utils.holistic_utils import extract_holistic_keypoints
from utils.inference import GlossDecoder, SignClassifier
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def draw_landmarks(image, results, draw_face=True):
    if draw_face and results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def main():
    parser = argparse.ArgumentParser(description="Real-time holistic sign translation.")
    parser.add_argument('--profile', default=None, help="low | high | experimental (default: $PROFILE or config default)")
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--keras', action='store_true', help="Use the Keras model even when a TFLite model exists")
    args = parser.parse_args()

    profile = get_profile(args.profile)
    classifier = SignClassifier(profile['name'], prefer_tflite=not args.keras)
    if not classifier.loaded:
        logger.error(f"No model for profile '{profile['name']}'. Run: python ml/train_model.py --profile {profile['name']}")
        return
    decoder = GlossDecoder(classifier.labels, threshold=args.threshold)
    seq_len = classifier.sequence_length

    cap = cv2.VideoCapture(args.camera)
    cam = profile['camera']
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam['height'])
    cap.set(cv2.CAP_PROP_FPS, cam['fps'])
    # Heavier pose model on the experimental (desktop) track only.
    complexity = 2 if profile['target'] == 'desktop' else 1

    sequence = []
    logger.info(f"--- Real-time inference: profile={profile['name']} camera={cam} T={seq_len} ---")
    with mp_holistic.Holistic(model_complexity=complexity, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            image = frame
            draw_landmarks(image, results, draw_face=profile['target'] == 'desktop')

            sequence = (sequence + [extract_holistic_keypoints(results)])[-seq_len:]
            if len(sequence) == seq_len:
                word, conf = decoder.update(classifier.predict(np.array(sequence)))
                if word:
                    cv2.putText(image, f"{word} {int(conf * 100)}%", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            h = image.shape[0]
            cv2.rectangle(image, (0, h - 80), (image.shape[1], h), (50, 50, 50), cv2.FILLED)
            cv2.putText(image, decoder.sentence, (20, h - 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(image, f"[{profile['name']}] [Q] Quit  [C] Clear", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Holistic Sign Translator", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                decoder.reset("Sentence Cleared.")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Session Ended.")


if __name__ == "__main__":
    main()
