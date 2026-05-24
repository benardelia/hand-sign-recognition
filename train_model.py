import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from utils.logger_config import setup_logger
from utils.normalization import normalize_landmarks, flip_landmarks

# Initialize Logger
logger = setup_logger(__name__)

class WebProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        acc = logs.get('categorical_accuracy', 0.0)
        loss = logs.get('loss', 0.0)
        val_acc = logs.get('val_categorical_accuracy', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        print(f"\nEPOCH_END:{epoch+1}:{acc:.4f}:{loss:.4f}:{val_acc:.4f}:{val_loss:.4f}", flush=True)

# --- CONFIGURATION ---
DATA_PATH = os.path.join('Data', 'Landmarks') 
HOLISTIC_DATA_PATH = os.path.join('Data', 'Holistic_Landmarks')
STATIC_PATH = os.path.join('Data', 'Static')   
MODELS_PATH = 'Model'
SEQUENCE_LENGTH = 30 
# INPUT_SHAPE will be determined dynamically

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def process_sample(res, is_sequence=True):
    """
    Normalizes a sample and creates an augmented (flipped) version for 63-shape hand data.
    Passes through holistic (1662) data without the hand-specific normalization.
    """
    processed = []
    
    # Check if holistic data
    is_holistic = res.shape[-1] == 1662
    
    if is_sequence:
        if is_holistic:
            processed.append(res)
            # Add simple flip for holistic if needed later, but skip for now
        else:
            norm_seq = np.array([normalize_landmarks(frame) for frame in res])
            processed.append(norm_seq)
            flip_seq = np.array([flip_landmarks(frame) for frame in norm_seq])
            processed.append(flip_seq)
    else:
        if is_holistic:
            # Static holistic not supported yet, just pass through repeated
            seq = np.tile(res, (SEQUENCE_LENGTH, 1))
            processed.append(seq)
        else:
            norm_frame = normalize_landmarks(res)
            norm_seq = np.tile(norm_frame, (SEQUENCE_LENGTH, 1))
            processed.append(norm_seq)
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
    if os.path.exists(HOLISTIC_DATA_PATH):
        all_labels += [d for d in os.listdir(HOLISTIC_DATA_PATH) if d not in all_labels and os.path.isdir(os.path.join(HOLISTIC_DATA_PATH, d))]
    if os.path.exists(STATIC_PATH):
        all_labels += [d for d in os.listdir(STATIC_PATH) if d not in all_labels and os.path.isdir(os.path.join(STATIC_PATH, d))]
    
    all_labels = sorted(list(set(all_labels)))
    label_map = {label: i for i, label in enumerate(all_labels)}
    logger.info(f"Detected Labels: {all_labels}")
    
    input_shape = None
    
    # Helper to process and append
    def add_files_from_path(data_path, is_seq=True):
        nonlocal input_shape
        if not os.path.exists(data_path): return
        for label in all_labels:
            path = os.path.join(data_path, label)
            if not os.path.exists(path): continue
            files = [f for f in os.listdir(path) if f.endswith('.npy')]
            for file in files:
                res = np.load(os.path.join(path, file))
                
                # Determine input shape from the first valid file
                if input_shape is None:
                    if is_seq:
                        input_shape = (SEQUENCE_LENGTH, res.shape[-1])
                    else:
                        input_shape = (SEQUENCE_LENGTH, res.shape[0])
                
                # Filter by expected sequence length if it's sequence data
                if is_seq and res.shape[0] != SEQUENCE_LENGTH:
                    continue
                # Also filter by expected feature size
                if (is_seq and res.shape[-1] != input_shape[-1]) or (not is_seq and res.shape[0] != input_shape[-1]):
                    continue
 
                for s in process_sample(res, is_sequence=is_seq):
                    sequences.append(s)
                    labels.append(label_map[label])
 
    # 2. Load Sequence Data
    add_files_from_path(HOLISTIC_DATA_PATH, is_seq=True)
    if input_shape is None:
        add_files_from_path(DATA_PATH, is_seq=True)
 
    # 3. Load Static Data
    add_files_from_path(STATIC_PATH, is_seq=False)
 
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    with open(os.path.join(MODELS_PATH, 'labels.txt'), 'w') as f:
        f.write("\n".join(all_labels))
    
    return X, y, all_labels, input_shape
 
def build_model(num_classes, input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
 
def train():
    logger.info("--- Phase 4: Training Model ---")
    X, y, all_labels, input_shape = load_data()
    if len(X) == 0:
        logger.error("No data found to train on!")
        return
 
    logger.info(f"Determined Input Shape: {input_shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = build_model(len(all_labels), input_shape)
    
    logger.info(f"Dataset Size (with augmentation): {len(X)}")
    model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), callbacks=[WebProgressCallback()])
    
    model.save(os.path.join(MODELS_PATH, 'hand_model.h5'))
    logger.info("Training Complete with improved normalization!")
 
if __name__ == "__main__":
    train()
