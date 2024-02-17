import numpy as np
import math

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes


class MusicGenrePredictor:
    def __init__(self):
        self.root = None
        self.feature_names = ["age", "favorite_color"]
        self.label_name = "genre_of_music"

    def entropy(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def split_data(self, data, feature_index, threshold):
        left = [d for d in data if d[feature_index] <= threshold]
        right = [d for d in data if d[feature_index] > threshold]
        return left, right

    def find_best_split(self, data):
        best_entropy = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(len(data[0]) - 1):
            values = [d[feature_index] for d in data]
            unique_values = np.unique(values)
            thresholds = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

            for threshold in thresholds:
                left, right = self.split_data(data, feature_index, threshold)

                if len(left) == 0 or len(right) == 0:
                    continue

                left_entropy = self.entropy([d[-1] for d in left])
                right_entropy = self.entropy([d[-1] for d in right])

                combined_entropy = (len(left) / len(data)) * left_entropy + (len(right) / len(data)) * right_entropy

                if combined_entropy < best_entropy:
                    best_entropy = combined_entropy
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def build_tree(self, data):
        labels = [d[-1] for d in data]

        if len(set(labels)) == 1:
            return DecisionNode(value=labels[0])

        if len(data[0]) == 1:  # No more features to split on
            return DecisionNode(value=max(set(labels), key=labels.count))

        best_feature_index, best_threshold = self.find_best_split(data)

        if best_feature_index is None:
            return DecisionNode(value=max(set(labels), key=labels.count))

        left, right = self.split_data(data, best_feature_index, best_threshold)

        left_node = self.build_tree(left)
        right_node = self.build_tree(right)

        return DecisionNode(feature=best_feature_index, threshold=best_threshold, left=left_node, right=right_node)

    def train(self, data):
        self.root = self.build_tree(data)

    def predict(self, data_point):
        current_node = self.root

        while current_node.left:  # While not a leaf node
            if data_point[current_node.feature] <= current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right

        return current_node.value


# Toy data
data = [
    [25, "Blue", "Pop"],
    [30, "Green", "Rock"],
    [20, "Red", "Pop"],
    [35, "Blue", "Rock"],
    [40, "Green", "Rock"],
    [22, "Red", "Pop"],
    [28, "Blue", "Pop"],
    [32, "Green", "Rock"],
    [18, "Red", "Pop"],
    [38, "Blue", "Rock"],
    [27, "Red", "Pop"],
    [33, "Green", "Rock"],
    [23, "Blue", "Pop"],
    [29, "Green", "Rock"],
    [21, "Red", "Pop"],
    [37, "Blue", "Rock"],
    [26, "Red", "Pop"],
    [31, "Green", "Rock"],
    [19, "Blue", "Pop"],
    [34, "Red", "Rock"]
]
color_map = {"Red": 0, "Green": 1, "Blue": 2}
for d in data:
    d[1] = color_map[d[1]]


# Create and train the model
predictor = MusicGenrePredictor()
predictor.train(data)

# Test the model with some examples
print(predictor.predict([29, 1]))  # Output: Rock
print(predictor.predict([24, 2]))  # Output: Pop
