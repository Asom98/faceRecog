import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x_train, y_train):
        """
        Store the training data.
        """
        self.x_train = x_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        """
        Compute the Euclidean distance between two points.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, x_test):
        """
        Predict the class labels for the test set.
        """
        predictions = []
        for test_point in x_test:
            distances = [self._euclidean_distance(test_point, train_point) for train_point in self.x_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

    def score(self, x_test, y_test):
        """
        Calculate the accuracy of the predictions.
        """
        y_pred = self.predict(x_test)
        return np.mean(y_pred == y_test)
