# logistic_regression.py

import numpy as np

class LogisticRegression:
    def __init__(self, input_size, num_classes, learning_rate=0.1, regularization=0.001):
        self.W = np.random.randn(input_size, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        self.learning_rate = learning_rate
        self.regularization = regularization
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        cross_entropy_loss = -np.sum(y * np.log(y_hat + 1e-8)) / m
        l2_loss = self.regularization * np.sum(np.square(self.W)) / 2
        return cross_entropy_loss + l2_loss
    
    def compute_gradients(self, X, y, y_hat):
        m = X.shape[0]
        dW = np.dot(X.T, (y_hat - y)) / m + self.regularization * self.W
        db = np.sum(y_hat - y, axis=0, keepdims=True) / m
        return dW, db
    
    def train(self, X, y, epochs=1000):
        for i in range(epochs):
            z = np.dot(X, self.W) + self.b
            y_hat = self.softmax(z)
            loss = self.compute_loss(y, y_hat)
            dW, db = self.compute_gradients(X, y, y_hat)
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss}")
    
    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_hat = self.softmax(z)
        return np.argmax(y_hat, axis=1)
