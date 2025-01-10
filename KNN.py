import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, KFold

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
    
        ### K-Fold Cross-Validation ###
    def cross_validate_knn(X, y, k=7, num_folds=5):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_accuracies = []

        for train_index, val_index in kf.split(X):
            x_train, x_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Train and evaluate k-NN model 
            knn = KNN(k=k)
            knn.fit(x_train, y_train)

            y_val_pred = knn.predict(x_val)
            val_accuracy = np.mean(y_val == y_val_pred)
            fold_accuracies.append(val_accuracy)
            print(f"Validation Accuracy for fold: {val_accuracy}")

        avg_accuracy = np.mean(fold_accuracies)
        print(f"Average Cross-Validation Accuracy: {avg_accuracy}")
        return avg_accuracy
