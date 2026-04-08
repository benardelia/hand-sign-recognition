import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Initialize Hand Detector (limited to 1 hand for simpler classification)
detector = HandDetector(maxHands=1)

# Initialize Classifier with the trained TensorFlow Keras model
# Ensure 'Model/keras_model.h5' and 'Model/labels.txt' exist in the 'Model' folder
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Configuration variables
offset = 40      # Extra space to crop around the hand
imgSize = 300    # Target square size for classification input

# The labels corresponding to the classes trained in your model
labels = ["A", "B", "C"]

while True:
    # Capture camera frame
    success, img = cap.read()
    if not success:
        break
    
    # Store a copy of the frame for drawing the output text and boxes
    imgOutput = img.copy()
    
    # Detect hand
    hands, img = detector.findHands(img)
    
    if hands:
        # Get position of the hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create white background for centered crop
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Calculate safe cropping coordinates within image boundaries
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        
        # Execute the crop
        imgCrop = img[y1:y2, x1:x2]

        # Stop processing this frame if the crop is somehow empty
        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        # Image processing to match the input format expected by the classifier
        if aspectRatio > 1:
            # Vertical hand: Height is longer
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            
            # Center horizontally
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            
            # Get prediction using the square 'imgWhite'
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(f"Prediction: {prediction}, Index: {index}")

        else:
            # Horizontal hand: Width is longer
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            
            # Center vertically
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            
            # Get prediction using the square 'imgWhite'
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Draw visual feedback on the 'imgOutput' screen
        # 1. Background rectangle for the label text
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        
        # 2. Add the prediction label text
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        
        # 3. Draw the main bounding box around the hand
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Display the processed view for debugging
        cv2.imshow("ImageWhite", imgWhite)

    # Show the final recognition output
    cv2.imshow("Image", imgOutput)
    
    # Handle user interaction
    key = cv2.waitKey(1)
    
    # Press 'q' to exit
    if key == ord('q'):
        break