# logistic_regression.py

import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LogisticRegression:
    def __init__(self, input_size, num_classes, learning_rate=0.1, regularization=0.001):
        self.w = np.random.randn(input_size, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        self.learning_rate = learning_rate
        self.regularization = regularization

    def _compute_loss(self, y, y_hat):
        m = y.shape[0]
        cross_entropy_loss = -np.sum(y * np.log(y_hat + 1e-8)) / m
        l2_loss = self.regularization * np.sum(np.square(self.w)) / 2
        return cross_entropy_loss + l2_loss

    def _compute_gradients(self, x, y, y_hat):
        m = x.shape[0]
        d_w = np.dot(x.T, (y_hat - y)) / m + self.regularization * self.w
        db = np.sum(y_hat - y, axis=0, keepdims=True) / m
        return d_w, db

    def train(self, x, y, epochs=1000):
        for i in range(epochs):
            z = np.dot(x, self.w) + self.b
            y_hat = softmax(z)
            loss = self._compute_loss(y, y_hat)
            d_w, db = self._compute_gradients(x, y, y_hat)
            self.w -= self.learning_rate * d_w
            self.b -= self.learning_rate * db
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss}")

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        y_hat = softmax(z)
        return np.argmax(y_hat, axis=1)



from sklearn.model_selection import KFold

### K-Fold Cross-Validation ###
def cross_validate_model(x, y, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the model
        model = LogisticRegression(input_size=x_train.shape[1],
                                   num_classes=y_train.shape[1],
                                   learning_rate=0.1,
                                   regularization=0.001)
        model.train(x_train, y_train, epochs=1000)

        # Validate the model
        y_val_pred = model.predict(x_val)
        val_accuracy = np.mean(np.argmax(y_val, axis=1) == y_val_pred)
        fold_accuracies.append(val_accuracy)
        print(f"Validation Accuracy for fold: {val_accuracy}")

    avg_accuracy = np.mean(fold_accuracies)
    print(f"Average Cross-Validation Accuracy: {avg_accuracy}")
    return avg_accuracy
