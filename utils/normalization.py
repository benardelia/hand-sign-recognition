import numpy as np

def normalize_landmarks(lmList):
    """
    Normalizes a list of landmarks to be wrist-relative and scaled.
    
    Args:
        lmList: Can be (21, 3) for a single frame or (63,) flattened.
        
    Returns:
        A flattened NumPy array of 63 normalized coordinates.
    """
    # Reshape if flattened
    if isinstance(lmList, np.ndarray) and lmList.shape == (63,):
        lmList = lmList.reshape(21, 3)
    
    if not lmList or len(lmList) != 21:
        return np.zeros(63)

    # 1. Translation: Move wrist (index 0) to (0,0,0)
    wrist = np.array(lmList[0])
    normalized_list = np.array(lmList) - wrist

    # 2. Scaling: Normalize by the distance between wrist (0) and middle finger base (9)
    middle_finger_base = normalized_list[9]
    distance = np.linalg.norm(middle_finger_base)
    
    if distance > 1e-6: # Avoid division by zero
        normalized_list = normalized_list / distance

    return normalized_list.flatten()

def flip_landmarks(lmList):
    """
    Simulates the other hand by flipping the X-axis.
    
    Args:
        lmList: (21, 3) or (63,) landmarks.
        
    Returns:
        Flipped flattened landmarks.
    """
    if isinstance(lmList, np.ndarray) and lmList.shape == (63,):
        lmList = lmList.reshape(21, 3)
    
    flipped = np.array(lmList).copy()
    flipped[:, 0] = -flipped[:, 0] # Flip X
    
    return flipped.flatten()
