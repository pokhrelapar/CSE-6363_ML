import numpy as np


class DecisionTree:

    def __init__(self, criterion="gini", max_depth=5, min_samples_split=5, min_samples_leaf=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):

        if self.tree is None:
            raise ValueError("Tree not built")

        return np.array([self._walk(x, self.tree) for x in X])

    def _calculate_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)

        if self.criterion == "gini":
            return 1 - np.sum(probs**2)

        elif self.criterion == "entropy":
            eps = 1e-10
            return -np.sum(probs * np.log2(probs + eps))

        elif self.criterion == "misclassification":
            return 1 - np.max(probs)

        else:
            raise ValueError("Criterion not valid")

    def _calculate_information_gain(self, parent, left_child, right_child):
        weight_l, weight_r = len(left_child) / len(parent), len(right_child) / len(parent)

        parent_impurity = self._calculate_impurity(parent)

        left_child_impurity = self._calculate_impurity(left_child)
        right_child_impurity = self._calculate_impurity(right_child)

        return parent_impurity - (weight_l * left_child_impurity + weight_r * right_child_impurity)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        labels, counts = np.unique(y, return_counts=True)

        # stop building tree
        if depth >= self.max_depth or len(labels) == 1 or num_samples < self.min_samples_split:
            return {"label": labels[np.argmax(counts)]}

        # find which feature to split on
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return {"label": labels[np.argmax(counts)]}

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self._build_tree(X[right_idx], y[right_idx], depth + 1),
        }

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape

        best_gain = -np.inf
        best_feature, best_threshold = None, None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])

            for th in thresholds:
                left_idx = X[:, feature] <= th
                right_idx = ~left_idx

                if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
                    continue

                gain = self._calculate_information_gain(y, y[left_idx], y[right_idx])

                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, th

        return best_feature, best_threshold

    def _walk(self, x, node):
        if "label" in node:
            return node["label"]

        elif x[node["feature"]] <= node["threshold"]:
            return self._walk(x, node["left"])

        else:
            return self._walk(x, node["right"])
