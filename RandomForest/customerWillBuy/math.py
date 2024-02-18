import data as da
import numpy as np
import pandas as pd
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return {'prediction': np.argmax(np.bincount(y))}

        feature_index, threshold = self._find_best_split(X, y)

        if feature_index is None:
            return {'prediction': np.argmax(np.bincount(y))}

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature_index': feature_index, 'threshold': threshold,
                'left': left_subtree, 'right': right_subtree}
    def _find_best_split(self, X, y):
        # Simple heuristic: find the split that maximizes information gain
        best_feature_index = None
        best_threshold = None
        best_info_gain = -1

        for feature_index in range(X.shape[1]):
            thresholds = sorted(set(X[:, feature_index]))

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if len(y[left_mask]) > 0 and len(y[right_mask]) > 0:
                    info_gain = self._information_gain(y, y[left_mask], y[right_mask])
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_feature_index = feature_index
                        best_threshold = threshold

        return best_feature_index, best_threshold

    def _information_gain(self, parent, left_child, right_child):
        # Simple information gain calculation
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        entropy_parent = self._entropy(parent)
        entropy_left = self._entropy(left_child)
        entropy_right = self._entropy(right_child)

        return entropy_parent - (weight_left * entropy_left + weight_right * entropy_right)

    def _entropy(self, y):
        # Use numpy.bincount to get counts of unique elements in the array
        unique_classes, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, node):
        if 'prediction' in node:
            return node['prediction']

        if x[node['feature_index']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

class RandomForestModel:
    def __init__(self, data, num_trees=10, max_depth=None):
        self.data = data
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

        self.train_random_forest()

    def train_random_forest(self, random_state=42):
        for _ in range(self.num_trees):
            subset_indices = np.random.choice(len(self.data), len(self.data), replace=True)
            subset_data = self.data.iloc[subset_indices]

            X = subset_data[['Browsing_History', 'Time_Spent']].values
            y = subset_data['Purchase'].values

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, y)

            self.trees.append(tree)

    def predict(self, new_data):
        predictions = np.array([tree.predict(new_data[['Browsing_History', 'Time_Spent']].values) for tree in self.trees])
        return np.round(np.mean(predictions, axis=0))

# Usage
data = da.SyntheticData()
random_forest_model = RandomForestModel(data)

# Create new synthetic data for prediction
new_data = da.predictionData(10)

# Make predictions
predictions = random_forest_model.predict(new_data)

# Display the predictions
print('\nnew data:')
print(new_data)
print('\nPredictions for the new data:')
print(predictions)
