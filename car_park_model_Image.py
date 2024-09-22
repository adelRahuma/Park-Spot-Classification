import os
import cv2
import pickle
import numpy as np
from skimage.transform import resize

# Load the saved SVM model
with open('model.p', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load parking slot positions from 'CarParkPoints'
try:
    with open('CarParkPoints', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# Define the width and height of each parking slot
width, height = 48, 21


def extract_features(image):
    resized_img = resize(image, (15, 15,3))
    features = resized_img.flatten()
    return features

def mouseClick(event, x, y, flags, params):
    """
    Mouse callback function to add or remove parking slots.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)

    with open('CarParkPoints', 'wb') as f:
        pickle.dump(posList, f)

# Path to the image
image_path = os.path.join(".","Images","Park.jpg")
 
while True:
    # Read the image
    img = cv2.imread(image_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    empty_count = 0
    not_empty_count = 0
    for pos in posList:
        x, y = pos
        # Crop the image around the parking slot
        crop_img = grayscale_img[y:y + height, x:x + width]
        # cv2.imshow('crop_img',crop_img)
        # cv2.waitKey(0)
        
        # Extract features from the cropped image
        features = extract_features(crop_img)
        features = np.array([features])  # Reshape to match input format for the model

        # Predict using the SVM model
        prediction = svm_model.predict(features)[0]

        # Assign color based on model prediction
        if prediction == 0:  # 0 represents 'empty'
            color = (0, 255, 0)  # Green for empty
            empty_count += 1 
        else:  # 1 represents 'not_empty'
            color = (0, 0, 255)  # Red for not empty
            not_empty_count +=1
            

        # Draw rectangle around the parking slot on the image
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2)

    # Display the image with marked parking slots
    cv2.putText(img, f'Empty Slots: {empty_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Parking Slot Detection', img)
    cv2.setMouseCallback('Parking Slot Detection', mouseClick)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
