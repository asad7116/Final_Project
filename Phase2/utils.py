import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """
    Preprocess the image:
    1. Resize to 8x8 (or any suitable size).
    2. Convert to grayscale.
    3. Normalize pixel values to [0, 1].
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    resized_image = cv2.resize(image, (8, 8))  # Resize to 8x8
    normalized_image = resized_image / 255.0  # Normalize to [0, 1]
    return normalized_image.flatten()  # Flatten into a 1D array

def load_data(directory, label):
    """
    Load images from the directory and return them as features with labels.
    """
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(directory, filename)
            image = preprocess_image(filepath)
            features.append(image)
            labels.append(label)
    return np.array(features), np.array(labels)
