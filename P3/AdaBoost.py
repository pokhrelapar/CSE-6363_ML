import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from copy import deepcopy

from DecisionTree import DecisionTree


class AdaBoost:
    def __init__(self, weak_learner=DecisionTree, num_learners=50, learning_rate=1.0):
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.alphas = []  #  alphas of each weak learner
        self.learners = []  #  weak learners (trained models)
        self.errors = []

    def fit(self, X, y):
        num_samples = X.shape[0]

        # Initialize sample weights (uniform distribution)
        sample_weights = np.ones(num_samples) / num_samples
        # print("initial", sample_weights)

        for m in range(self.num_learners):
            # Train the weak learner using the weighted dataset
            learner = deepcopy(self.weak_learner())
            learner.fit(X, y)

            # Make predictions for the current weak learner
            predictions = learner.predict(X)

            # error for learner
            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            # If error is 0, stop early
            if error == 0:
                break

            # alpha of weak learner
            eps = 1e-10
            alpha = self.learning_rate * 0.5 * np.log((1 - error + eps) / (error + eps))

            # Save the learner, alphas and error values
            self.learners.append(learner)
            self.alphas.append(alpha)
            self.errors.append(error)

            # print("alphas", self.alphas)

            # Update the sample weights
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])

        for alpha, learner in zip(self.alphas, self.learners):
            predictions = learner.predict(X)
            final_predictions += alpha * predictions

        return np.sign(final_predictions)
