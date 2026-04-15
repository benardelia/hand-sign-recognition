import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils.logger_config import setup_logger
from utils.normalization import normalize_landmarks, flip_landmarks

# Initialize Logger
logger = setup_logger(__name__)

# --- CONFIGURATION ---
DATA_PATH = os.path.join('Data', 'Landmarks') 
STATIC_PATH = os.path.join('Data', 'Static')   
MODELS_PATH = 'Model'
SEQUENCE_LENGTH = 30 
NUM_LANDMARKS = 21
NUM_COORDS = 3
INPUT_SHAPE = (SEQUENCE_LENGTH, NUM_LANDMARKS * NUM_COORDS)

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def process_sample(res, is_sequence=True):
    """
    Normalizes a sample and creates an augmented (flipped) version.
    Returns: List of processed sequences.
    """
    processed = []
    
    if is_sequence:
        # Normalize every frame in the sequence
        norm_seq = np.array([normalize_landmarks(frame) for frame in res])
        processed.append(norm_seq)
        
        # Augmentation: Horizontal Flip
        flip_seq = np.array([flip_landmarks(frame) for frame in norm_seq])
        processed.append(flip_seq)
    else:
        # Static: Normalize, then repeat to 30 frames
        norm_frame = normalize_landmarks(res)
        norm_seq = np.tile(norm_frame, (SEQUENCE_LENGTH, 1))
        processed.append(norm_seq)
        
        # Static Augmentation: Flip
        flip_frame = flip_landmarks(norm_frame)
        flip_seq = np.tile(flip_frame, (SEQUENCE_LENGTH, 1))
        processed.append(flip_seq)
        
    return processed

def load_data():
    sequences, labels = [], []
    all_labels = []
    
    # 1. Detect all labels
    if os.path.exists(DATA_PATH):
        all_labels += [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    if os.path.exists(STATIC_PATH):
        all_labels += [d for d in os.listdir(STATIC_PATH) if d not in all_labels and os.path.isdir(os.path.join(STATIC_PATH, d))]
    
    all_labels = sorted(list(set(all_labels)))
    label_map = {label: i for i, label in enumerate(all_labels)}
    logger.info(f"Detected Labels: {all_labels}")
    
    # 2. Load Sequence Data
    if os.path.exists(DATA_PATH):
        for label in all_labels:
            path = os.path.join(DATA_PATH, label)
            if not os.path.exists(path): continue
            files = [f for f in os.listdir(path) if f.endswith('.npy')]
            for file in files:
                res = np.load(os.path.join(path, file))
                if res.shape == INPUT_SHAPE:
                    for s in process_sample(res, is_sequence=True):
                        sequences.append(s)
                        labels.append(label_map[label])

    # 3. Load Static Data
    if os.path.exists(STATIC_PATH):
        for label in all_labels:
            path = os.path.join(STATIC_PATH, label)
            if not os.path.exists(path): continue
            files = [f for f in os.listdir(path) if f.endswith('.npy')]
            for file in files:
                res = np.load(os.path.join(path, file))
                if res.shape == (63,):
                    for s in process_sample(res, is_sequence=False):
                        sequences.append(s)
                        labels.append(label_map[label])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    with open(os.path.join(MODELS_PATH, 'labels.txt'), 'w') as f:
        f.write("\n".join(all_labels))
    
    return X, y, all_labels

def build_model(num_classes):
    model = Sequential()
    # BatchNormalization could be added here, but our custom normalization handles it.
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=INPUT_SHAPE))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def train():
    logger.info("--- Quality Improvement: Re-training Model with Normalization ---")
    X, y, all_labels = load_data()
    if len(X) == 0:
        logger.error("No data found to train on!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = build_model(len(all_labels))
    
    logger.info(f"Dataset Size (with augmentation): {len(X)}")
    model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))
    
    model.save(os.path.join(MODELS_PATH, 'hand_model.h5'))
    logger.info("Training Complete with improved normalization!")

if __name__ == "__main__":
    train()
