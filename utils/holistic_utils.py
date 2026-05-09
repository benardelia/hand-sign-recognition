import numpy as np

def extract_holistic_keypoints(results):
    """
    Extracts and flattens all landmarks from a MediaPipe Holistic result into a single 1D array.
    Shape:
    - Pose: 33 landmarks * 4 coords (x,y,z,visibility) = 132
    - Face: 468 landmarks * 3 coords (x,y,z) = 1404
    - Left Hand: 21 landmarks * 3 coords (x,y,z) = 63
    - Right Hand: 21 landmarks * 3 coords (x,y,z) = 63
    - Total: 1662 values
    """
    # Pose: 33 * 4
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Face: 468 * 3
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)

    # Left Hand: 21 * 3
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Right Hand: 21 * 3
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])
