import numpy as np
import os
from sklearn.model_selection import train_test_split
from ann import SimpleANN as ANN  # Ensure ANN implementation is in the same directory
from utils import load_data, preprocess_image

# Parameters
learning_rate = 0.1  # Increased from 0.01
hidden_layer_size = 16  # Increased from 8
epochs = 2000  # Increased from 1000

# Load Data
X_hearts, y_hearts = load_data(r"D:\Semester_5\AI\Final_Project\dataSet\Hearts", label=0)
X_diamonds, y_diamonds = load_data(r"D:\Semester_5\AI\Final_Project\dataSet\Diamonds", label=1)

# Combine and Split Data
X = np.vstack((X_hearts, X_diamonds))
y = np.hstack((y_hearts, y_diamonds))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize ANN
ann = ANN(input_size=X_train.shape[1], hidden_size=hidden_layer_size, output_size=2, learning_rate=learning_rate)
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Assuming binary classification
y_train_one_hot = one_hot_encode(y_train, num_classes=2)

# Now, pass y_train_one_hot to ANN
loss = ann.train(X_train, y_train_one_hot)

# Training
losses = []
for epoch in range(epochs):
    loss = ann.train(X_train, y_train)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Debug: Training Accuracy
predictions_train = ann.predict(X_train)
train_accuracy = np.sum(predictions_train == y_train) / y_train.size * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")

# Debug: Test Accuracy
predictions_test = ann.predict(X_test)
test_accuracy = np.sum(predictions_test == y_test) / y_test.size * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save Model
np.save("weights_input_hidden.npy", ann.weights_input_hidden)
np.save("weights_hidden_output.npy", ann.weights_hidden_output)
