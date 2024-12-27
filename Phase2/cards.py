import numpy as np
import cv2
import os

# ============================
# Step 1: Dataset Preparation
# ============================
def load_dataset(folder_path):
    """
    Loads dataset images, converts them to grayscale, and resizes them to a fixed size.
    Args:
        folder_path (str): Path to the dataset folder.

    Returns:
        data (list): List of image arrays.
        labels (list): Corresponding labels for the images.
    """
    data = []
    labels = []
    classes = os.listdir(folder_path)

    for label, class_name in enumerate(classes):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Resize to 64x64
                    data.append(img.flatten())  # Flatten the image
                    labels.append(label)

    return np.array(data), np.array(labels)

# Load the dataset
train_data, train_labels = load_dataset(r"D:\Semester_5\AI\Final_Project\dataSet\train")
test_data, test_labels = load_dataset(r"D:\Semester_5\AI\Final_Project\dataSet\test")

# ============================
# Step 2: ANN Implementation
# ============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class ANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output

    def backward(self, X, y):
        # Calculate error
        error = self.output - y

        # Output to hidden layer gradients
        d_output = error * sigmoid_derivative(self.output)
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, d_output)
        self.bias_output -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)

        # Hidden to input layer gradients
        hidden_error = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden = hidden_error * sigmoid_derivative(self.hidden_output)
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, d_hidden)
        self.bias_hidden -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = np.mean((y - self.output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)

# One-hot encoding for labels
def one_hot_encode(labels, num_classes):
    if np.any(labels >= num_classes):
        raise ValueError(f"Labels contain values outside the valid range 0 to {num_classes-1}")
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded

# Prepare data
train_labels -= 1
test_labels -= 1

num_classes = len(np.unique(train_labels))
print(f"Number of classes: {num_classes}")
print(f"Unique train labels: {np.unique(train_labels)}")
print(f"Unique test labels: {np.unique(test_labels)}")

# Check for invalid labels
invalid_train_labels = train_labels[train_labels >= num_classes]
invalid_test_labels = test_labels[test_labels >= num_classes]
if len(invalid_train_labels) > 0 or len(invalid_test_labels) > 0:
    print(f"Invalid train labels: {invalid_train_labels}")
    print(f"Invalid test labels: {invalid_test_labels}")
    raise ValueError("Labels contain values outside the valid range")

train_labels_encoded = one_hot_encode(train_labels, num_classes)
test_labels_encoded = one_hot_encode(test_labels, num_classes)

# Initialize and train the ANN
input_size = train_data.shape[1]
hidden_size = 64
output_size = num_classes
learning_rate = 0.1

ann = ANN(input_size, hidden_size, output_size, learning_rate)
ann.train(train_data, train_labels_encoded, epochs=1000)

# ============================
# Step 3: Evaluation
# ============================
# Evaluate on test data
def evaluate(ann, test_data, test_labels):
    predictions = ann.predict(test_data)
    accuracy = np.mean(predictions == test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

evaluate(ann, test_data, test_labels)
