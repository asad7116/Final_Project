from ann import SimpleANN as ANN
from preprocess import preprocess_image  # Import from the updated preprocess.py
import numpy as np
import os

# Path to the image
test_image_path = r"D:\Semester_5\AI\Final_Project\dataset1\test\ten of spades\1.jpg"

# Check if the image exists
if not os.path.exists(test_image_path):
    print(f"Error: File at {test_image_path} does not exist.")
else:
    # Print path for debugging
    print(f"Loading image from: {test_image_path}")
    
    # Load and preprocess the image
    test_image_features = preprocess_image(test_image_path, target_size=(64, 64)).reshape(1, -1)  # Update target size to 64x64

    # Debug: Check feature dimensions
    print(f"Feature vector shape: {test_image_features.shape}")
    assert test_image_features.shape[1] == 64 * 64, "Input size mismatch!"

    # Load Model Weights
    weights_input_hidden1 = np.load("weights_input_hidden1.npy")
    weights_hidden1_hidden2 = np.load("weights_hidden1_hidden2.npy")
    weights_hidden2_output = np.load("weights_hidden2_output.npy")

    # Initialize ANN
    ann = ANN(input_size=64*64, hidden_size1=512, hidden_size2=512, output_size=4)  # Match dimensions and update output size
    ann.weights_input_hidden1 = weights_input_hidden1
    ann.weights_hidden1_hidden2 = weights_hidden1_hidden2
    ann.weights_hidden2_output = weights_hidden2_output

    # Predict
    raw_output = ann.predict(test_image_features)
    predicted_class = np.argmax(raw_output)
    label_mapping = {0: "Hearts", 1: "Diamonds", 2: "Clubs", 3: "Spades"}  # Remove Joker
    print(f"Predicted Class: {label_mapping[predicted_class]}")

    # Debug: Raw Output
    print(f"Raw Output: {raw_output}")
