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
        
        # Handle different data shapes (Static vs Sequence)
        if data.ndim == 1:
            data_type = "Static (Single Frame)"
            num_frames = 1
            num_coords = data.shape[0]
        else:
            data_type = "Sequence"
            num_frames = data.shape[0]
            num_coords = data.shape[1]

        print("\n" + "="*50)
        print(f"FILE INFO:")
        print(f" - Type: {data_type}")
        print(f" - Shape: {data.shape}")
        print(f" - Frames: {num_frames}")
        print(f" - Coordinates per frame: {num_coords}")
        
        print("\nDATA PREVIEW:")
        if data_type == "Static (Single Frame)":
            print(f" First 10 values: {data[:10]}")
        else:
            print(f" Frame 1 (First 10 values): {data[0, :10]}")
            if num_frames > 1:
                print(f" Frame {num_frames} (First 10 values): {data[-1, :10]}")
        
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to load .npy file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        preview(sys.argv[1])
    else:
        logger.warning("No file path provided.")
        print("Usage: python preview_npy.py <path_to_npy_file>")
