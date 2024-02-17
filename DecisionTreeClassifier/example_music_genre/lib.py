from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class MusicGenrePredictor:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.feature_names = ["age", "favorite_color"]
        self.label_name = "genre_of_music"
        self.encoder = OneHotEncoder()

    def train(self, data):
        features = np.array([[d["age"], d["favorite_color"]] for d in data])
        labels = np.array([d[self.label_name] for d in data])

        # One-hot encode the categorical feature (favorite_color)
        features_encoded = self.encoder.fit_transform(features[:, 1].reshape(-1, 1))

        # Concatenate age and encoded favorite_color
        features_final = np.hstack((features[:, 0].reshape(-1, 1), features_encoded.toarray()))

        self.model.fit(features_final, labels)

    def predict(self, age, favorite_color):
        # Encode the favorite_color
        color_encoded = self.encoder.transform([[favorite_color]])

        # Concatenate age and encoded favorite_color
        features = np.hstack(([[age]], color_encoded.toarray()))

        return self.model.predict(features)[0]


# Toy data
data = [
    {"age": 25, "favorite_color": "Blue", "genre_of_music": "Pop"},
    {"age": 30, "favorite_color": "Green", "genre_of_music": "Rock"},
    {"age": 20, "favorite_color": "Red", "genre_of_music": "Pop"},
    {"age": 35, "favorite_color": "Blue", "genre_of_music": "Rock"},
    {"age": 40, "favorite_color": "Green", "genre_of_music": "Rock"},
    {"age": 22, "favorite_color": "Red", "genre_of_music": "Pop"},
    {"age": 28, "favorite_color": "Blue", "genre_of_music": "Pop"},
    {"age": 32, "favorite_color": "Green", "genre_of_music": "Rock"},
    {"age": 18, "favorite_color": "Red", "genre_of_music": "Pop"},
    {"age": 38, "favorite_color": "Blue", "genre_of_music": "Rock"},
    {"age": 27, "favorite_color": "Red", "genre_of_music": "Pop"},
    {"age": 33, "favorite_color": "Green", "genre_of_music": "Rock"},
    {"age": 23, "favorite_color": "Blue", "genre_of_music": "Pop"},
    {"age": 29, "favorite_color": "Green", "genre_of_music": "Rock"},
    {"age": 21, "favorite_color": "Red", "genre_of_music": "Pop"},
    {"age": 37, "favorite_color": "Blue", "genre_of_music": "Rock"},
    {"age": 26, "favorite_color": "Red", "genre_of_music": "Pop"},
    {"age": 31, "favorite_color": "Green", "genre_of_music": "Rock"},
    {"age": 19, "favorite_color": "Blue", "genre_of_music": "Pop"},
    {"age": 34, "favorite_color": "Red", "genre_of_music": "Rock"}
]

# Create and train the model
predictor = MusicGenrePredictor()
predictor.train(data)

# Test the model with some examples
print(predictor.predict(29, "Green"))  # Output: Rock
print(predictor.predict(24, "Blue"))  # Output: Pop
