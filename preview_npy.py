import numpy as np
import os
import sys
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def preview(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    try:
        data = np.load(file_path)
        logger.info(f"Successfully loaded: {os.path.basename(file_path)}")
        
        # Detect Shape
        if data.ndim == 1:
            data_type = "Static (Single Frame)"
            num_frames = 1
            num_coords = data.shape[0]
            max_val = np.max(np.abs(data))
        else:
            data_type = "Sequence"
            num_frames = data.shape[0]
            num_coords = data.shape[1]
            max_val = np.max(np.abs(data))

        # Detect Normalization
        # Raw pixel values are usually 100-1000+, normalized are ~0-2.
        is_normalized = max_val < 5

        print("\n" + "="*50)
        print(f"FILE INFO:")
        print(f" - Type: {data_type}")
        print(f" - Status: {'[Normalized]' if is_normalized else '[Raw / Unnormalized]'}")
        print(f" - Shape: {data.shape}")
        print(f" - Frames: {num_frames}")
        
        print("\nDATA PREVIEW:")
        if data_type == "Static (Single Frame)":
            print(f" First 6 values: {data[:6]}")
        else:
            print(f" Frame 1 (First 6 values): {data[0, :6]}")
        
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to load .npy file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        preview(sys.argv[1])
    else:
        logger.warning("No file path provided.")
        print("Usage: python preview_npy.py <path_to_npy_file>")
