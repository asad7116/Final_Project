import numpy as np
from ann import ANN
# from utils import preprocess_image
from utils import load_data, preprocess_image

# Load Model Weights
weights_input_hidden = np.load("weights_input_hidden.npy")
weights_hidden_output = np.load("weights_hidden_output.npy")

# Initialize ANN
ann = ANN(input_size=64, hidden_layer_size=16, output_size=2)  # Match dimensions
ann.weights_input_hidden = weights_input_hidden
ann.weights_hidden_output = weights_hidden_output

# Test Single Image
test_image_path = r"D:\Semester_5\AI\Final_Project\dataSet\Diamonds\test_diamond.jpg"
test_image_features = preprocess_image(test_image_path).reshape(1, -1)

# Predict
raw_output = ann.forward(test_image_features)
predicted_class = np.argmax(raw_output)
label_mapping = {0: "Hearts", 1: "Diamonds"}
print(f"Predicted Class: {label_mapping[predicted_class]}")

# Debug: Raw Output
print(f"Raw Output: {raw_output}")
