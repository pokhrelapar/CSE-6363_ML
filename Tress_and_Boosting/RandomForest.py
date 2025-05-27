import numpy as np
from copy import deepcopy
from DecisionTree import DecisionTree


class RandomForest:

    def __init__(self, classifier, num_trees=5, min_features=3):
        self.classifier = classifier
        self.num_trees = num_trees
        self.min_features = min_features
        self.trees = []
        self.features_set = []

    def fit(self, X, y):
        num_samples, num_features = X.shape

        if self.min_features > num_features or self.min_features < 1:
            raise ValueError("Error in number of features.")

        for i in range(self.num_trees):

            # sampling with replacement
            sample_idxs = np.random.choice(num_samples, size=num_samples, replace=True)
            x_sample, y_sample = X[sample_idxs], y[sample_idxs]

            # features selection
            num_selected_ftrs = np.random.randint(self.min_features, num_features + 1)
            selected_ftrs = np.random.choice(num_features, size=num_selected_ftrs, replace=False)
            print("Selected feature indices", selected_ftrs)

            self.features_set.append(selected_ftrs)

            tree = self.classifier()
            tree.fit(x_sample[:, selected_ftrs], y_sample)

            self.trees.append(tree)

    def predict(self, X):
        # Get predictions from all trees
        all_preds = np.zeros((self.num_trees, X.shape[0]), dtype=int)

        for i, (tree, features) in enumerate(zip(self.trees, self.features_set)):
            all_preds[i, :] = tree.predict(X[:, features])

        final_preds = []
        for i in range(X.shape[0]):
            sample_votes = all_preds[:, i]
            # common
            most_common = np.unique(sample_votes, return_counts=True)
            most_common = most_common[0][np.argmax(most_common[1])]
            final_preds.append(most_common)

        return np.array(final_preds)
