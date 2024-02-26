import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import data as da

class SvmClassifier:
    def __init__(self, X, y):
        self.model = SVC(kernel='linear')
        self.train(X, y)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X_new):
        return self.model.predict(X_new)


X_train, y_train = da.SyntheticData()

svc = SvmClassifier(X_train, y_train)

X_new, y_new = da.SyntheticData(1)

predicted_label = svc.predict(X_new)
print("real:", y_new)
print("prediction:", predicted_label)

