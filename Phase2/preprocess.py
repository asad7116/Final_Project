import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(64, 64)):  # Update target size to 64x64
    """
    Preprocess the image:
    1. Read the image in grayscale.
    2. Resize to the target size (e.g., 64x64).
    3. Normalize pixel values to [0, 1].
    4. Flatten the image into a 1D array.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Resize image to the desired size (e.g., 64x64)
    resized_image = cv2.resize(image, target_size)
    
    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image / 255.0
    
    # Flatten the image to a 1D vector
    return normalized_image.flatten()

def load_data(directory, label, target_size=(64, 64)):  # Update target size to 64x64
    """
    Load and preprocess all images in a given directory.
    Args:
    - directory: Directory containing image files.
    - label: The label for all images in this directory.
    - target size: The target size to which images should be resized (default is 64x64).
    
    Returns:
    - A tuple of features (numpy array) and labels (numpy array).
    """
    features = []
    labels = []
    
    # Loop through all image files in the directory
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Fix the syntax error here
            filepath = os.path.join(directory, filename)
            # Preprocess each image
            image = preprocess_image(filepath, target_size)
            features.append(image)
            labels.append(label)
    
    # Return the feature and label arrays
    return np.array(features), np.array(labels)

def load_and_preprocess_data(data_dir, target_size=(64, 64)):  # Update target size to 64x64
    """
    Load and preprocess data from all the 4 classes (Hearts, Diamonds, Clubs, Spades).
    """
    # Load data for each class
    X_hearts, y_hearts = load_data(os.path.join(data_dir, "Hearts"), label=0, target_size=target_size)
    X_diamonds, y_diamonds = load_data(os.path.join(data_dir, "Diamonds"), label=1, target_size=target_size)
    X_clubs, y_clubs = load_data(os.path.join(data_dir, "Clubs"), label=2, target_size=target_size)
    X_spades, y_spades = load_data(os.path.join(data_dir, "Spades"), label=3, target_size=target_size)
    # X_joker, y_joker = load_data(os.path.join(data_dir, "Joker"), label=4, target_size=target_size)  # Comment out Joker
    
    # Combine all the data
    X = np.vstack((X_hearts, X_diamonds, X_clubs, X_spades))  # Remove Joker
    y = np.hstack((y_hearts, y_diamonds, y_clubs, y_spades))  # Remove Joker
    
    return X, y
