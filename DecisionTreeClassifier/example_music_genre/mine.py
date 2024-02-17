from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np




class predictMusicGenre:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.features_names = ['age' , 'favorite_color']
        self.label_name = 'genre_of_music'
        self.encoder = OneHotEncoder()

    def train(self, data):
        features = np.array([[d['age'], d['favorite_color']] for d in data])
        labels = np.array([[d['genre_of_music']] for d in data])


        color_encode = self.encoder.fit_transform(features[:, 1].reshape(-1, 1))


        final_features = np.hstack((features[:, 0].reshape(-1, 1), color_encode.toarray()))

        self.model.fit(final_features, labels)

    def predict(self, age, favorite_color):

        color_encode = self.encoder.transform([[favorite_color]])
        features = np.hstack(([[age]], color_encode.toarray()))

        return self.model.predict(features)[0]


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

predictor = predictMusicGenre()

predictor.train(data)

print(predictor.predict('23', 'Blue'))
