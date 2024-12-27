from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from ann import SimpleANN  # Ensure this matches your ANN class name
from preprocess import load_data
import os

# Load data for all 4 classes
X_hearts, y_hearts = load_data("D:/Semester_5/AI/Final_Project/dataSet/train/Hearts", label=0)
X_diamonds, y_diamonds = load_data("D:/Semester_5/AI/Final_Project/dataSet/train/Diamonds", label=1)
X_clubs, y_clubs = load_data("D:/Semester_5/AI/Final_Project/dataSet/train/Clubs", label=2)
X_spades, y_spades = load_data("D:/Semester_5/AI/Final_Project/dataSet/train/Spades", label=3)
# X_joker, y_joker = load_data("D:/Semester_5/AI/Final_Project/dataSet/Jokers", label=4)  # Comment out Joker

# Combine data
X = np.vstack((X_hearts, X_diamonds, X_clubs, X_spades))  # Remove Joker
y = np.hstack((y_hearts, y_diamonds, y_clubs, y_spades))  # Remove Joker

# One-hot encode labels for 4 classes
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Update input size to 64x64
input_size = 64 * 64

# Initialize ANN with additional hidden layer
hidden_layer_size1 = 512  # First hidden layer size
hidden_layer_size2 = 512  # Second hidden layer size
output_size = 4  # 4 classes
ann = SimpleANN(input_size=input_size, hidden_size1=hidden_layer_size1, hidden_size2=hidden_layer_size2, output_size=output_size)

# Set the learning rate (e.g., 0.1)
learning_rate = 0.01  # Decrease learning rate

# Train the model
ann.train(X_train, y_train, epochs=5000, learning_rate=learning_rate)  # Reduce epochs

# Evaluate the model
y_pred = ann.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert one-hot encoded predictions to class labels
y_test_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels

accuracy = np.mean(y_pred_labels == y_test_labels) * 100
print(f"Accuracy on Test Set: {accuracy:.2f}%")
# Save weights
np.save("weights_input_hidden1.npy", ann.weights_input_hidden1)
np.save("weights_hidden1_hidden2.npy", ann.weights_hidden1_hidden2)
np.save("weights_hidden2_output.npy", ann.weights_hidden2_output)
