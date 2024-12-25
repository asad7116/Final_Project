import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to a fixed size
    resized_image = cv2.resize(image, (64, 64))  # Example size, modify as needed
    
    # Flatten the image into a 1D array
    features = resized_image.flatten()
    return features

def load_data(directory, label):
    data = []
    labels = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        features = preprocess_image(file_path)
        data.append(features)
        labels.append(label)
    return np.array(data), np.array(labels)
