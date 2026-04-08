import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize Webcam
# 0 is usually the default internal camera
cap = cv2.VideoCapture(0)

# Initialize Hand Detector
# maxHands=2 allows detection of up to two hands simultaneously
detector = HandDetector(maxHands=2)

# Configuration variables
offset = 40      # Extra space to crop around the hand
imgSize = 300    # Final size of the square image (300x300)
folder = "Data/C" # Directory where images will be saved
counter = 0      # Counter for saved images

# Ensure the directory exists
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    # Capture frame-by-frame from webcam
    success, img = cap.read()
    if not success:
        break
    
    # Detect hands in the frame
    # draw=True shows the skeleton on the original 'img'
    hands, img = detector.findHands(img)
    
    if hands:
        # Get data for the first hand detected
        hand = hands[0]
        x, y, w, h = hand['bbox'] # Bounding box coordinates and dimensions

        # Create a white background of fixed size
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Calculate cropping boundaries with offset
        # max/min ensure we don't crop outside the actual image frame
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        
        # Crop the hand area from the original image
        imgCrop = img[y1:y2, x1:x2]

        # Check if crop is valid (not empty)
        if imgCrop.size == 0:
            continue

        # Resize logic to fit the hand into the 300x300 square without distortion
        aspectRatio = h / w

        if aspectRatio > 1:
            # If hand is taller than wide
            k = imgSize / h
            wCal = math.ceil(k * w) # Calculate equivalent width
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            
            # Center the resized image horizontally on the white background
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            # If hand is wider than tall
            k = imgSize / w
            hCal = math.ceil(k * h) # Calculate equivalent height
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            
            # Center the resized image vertically on the white background
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Show the secondary window
        cv2.imshow("ImageWhite", imgWhite)

    # Show the main camera feed
    cv2.imshow("Image", img)
    
    # Check for key presses
    key = cv2.waitKey(1)
    
    # Press 'q' to quit
    if key == ord("q"):
        break
    
    # Press 's' to save the current 'imgWhite' to the folder
    if key == ord("s"):
        counter += 1
        # Save image with a unique timestamp to avoid overwriting
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")