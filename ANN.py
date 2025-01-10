import numpy as np

### Utility Functions ###

# ReLU Activation
def relu(z):
    return np.maximum(0, z)

# Derivative of ReLU
def relu_derivative(z):
    return (z > 0).astype(float)

# Softmax Activation
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Avoid overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Categorical Cross-Entropy Loss
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # Avoid log(0)
    return loss

# Initialize Parameters
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    parameters = {
        "W1": np.random.randn(input_size, hidden_size) * 0.01,
        "b1": np.zeros((1, hidden_size)),
        "W2": np.random.randn(hidden_size, output_size) * 0.01,
        "b2": np.zeros((1, output_size)),
    }
    return parameters

# Forward Propagation
def forward_propagation(X, parameters):
    Z1 = np.dot(X, parameters["W1"]) + parameters["b1"]
    A1 = relu(Z1)
    Z2 = np.dot(A1, parameters["W2"]) + parameters["b2"]
    A2 = softmax(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# Backward Propagation
def backward_propagation(X, y, cache, parameters):
    m = X.shape[0]

    # Gradients for output layer
    dZ2 = cache["A2"] - y
    dW2 = np.dot(cache["A1"].T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Gradients for hidden layer
    dA1 = np.dot(dZ2, parameters["W2"].T)
    dZ1 = dA1 * relu_derivative(cache["Z1"])
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Update Parameters
def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    return parameters

### ANN Model ###
class ANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.parameters = initialize_parameters(input_size, hidden_size, output_size)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        for epoch in range(self.epochs):
            # Forward propagation
            y_pred, cache = forward_propagation(X, self.parameters)

            # Compute loss
            loss = cross_entropy_loss(y, y_pred)

            # Backward propagation
            gradients = backward_propagation(X, y, cache, self.parameters)

            # Update parameters
            self.parameters = update_parameters(self.parameters, gradients, self.learning_rate)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        y_pred, _ = forward_propagation(X, self.parameters)
        return np.argmax(y_pred, axis=1)


