class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = self.sigmoid(final_input)

            # Backpropagation
            error = y - final_output
            d_output = error * self.sigmoid_derivative(final_output)
            error_hidden = np.dot(d_output, self.weights_hidden_output.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_output)

            # Update weights and biases
            self.weights_hidden_output += np.dot(hidden_output.T, d_output) * 0.1
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * 0.1
            self.weights_input_hidden += np.dot(X.T, d_hidden) * 0.1
            self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * 0.1

            if epoch % 100 == 0:
                loss = np.mean(np.square(error))  # Loss for monitoring
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self.sigmoid(final_input)
        return np.argmax(final_output, axis=1)
