import numpy as np

class SimpleANN:
    def __init__(self, input_size, output_size, hidden_size1=512, hidden_size2=512):
        # Initialize weights and biases
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size1) * 0.1  # Weights for input to first hidden layer
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2) * 0.1  # Weights for first hidden to second hidden layer
        self.weights_hidden2_output = np.random.randn(hidden_size2, output_size) * 0.1  # Weights for second hidden to output layer
        self.bias_hidden1 = np.zeros((1, hidden_size1))  # Bias for first hidden layer
        self.bias_hidden2 = np.zeros((1, hidden_size2))  # Bias for second hidden layer
        self.bias_output = np.zeros((1, output_size))  # Bias for output layer

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward pass
            hidden_input1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
            hidden_output1 = self.relu(hidden_input1)  # Activation for first hidden layer
            hidden_input2 = np.dot(hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden2
            hidden_output2 = self.relu(hidden_input2)  # Activation for second hidden layer
            final_input = np.dot(hidden_output2, self.weights_hidden2_output) + self.bias_output
            final_output = self.softmax(final_input)  # Activation for output layer

            # Backpropagation
            error = y - final_output
            d_output = error * final_output * (1 - final_output)  # Softmax derivative
            error_hidden2 = np.dot(d_output, self.weights_hidden2_output.T)
            d_hidden2 = error_hidden2 * self.relu_derivative(hidden_output2)  # Use ReLU derivative for second hidden layer
            error_hidden1 = np.dot(d_hidden2, self.weights_hidden1_hidden2.T)
            d_hidden1 = error_hidden1 * self.relu_derivative(hidden_output1)  # Use ReLU derivative for first hidden layer

            # Update weights and biases
            self.weights_hidden2_output += np.dot(hidden_output2.T, d_output) * learning_rate
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            self.weights_hidden1_hidden2 += np.dot(hidden_output1.T, d_hidden2) * learning_rate
            self.bias_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * learning_rate
            self.weights_input_hidden1 += np.dot(X.T, d_hidden1) * learning_rate
            self.bias_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * learning_rate

            if epoch % 100 == 0:
                loss = np.mean(np.square(error))  # Loss for monitoring
                accuracy = np.mean(np.argmax(final_output, axis=1) == np.argmax(y, axis=1)) * 100  # Calculate accuracy
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    def predict(self, X):
        hidden_input1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        hidden_output1 = self.relu(hidden_input1)  # Activation for first hidden layer
        hidden_input2 = np.dot(hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden2
        hidden_output2 = self.relu(hidden_input2)  # Activation for second hidden layer
        final_input = np.dot(hidden_output2, self.weights_hidden2_output) + self.bias_output
        final_output = self.softmax(final_input)  # Activation for output layer
        return final_output
