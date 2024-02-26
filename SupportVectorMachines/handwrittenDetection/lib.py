import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
q
class SVMClassifier:
    def __init__(self):
        self.model = SVC(kernel='linear')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


# Generate synthetic data
X_train, y_train = da.SyntheticData()

# Create SVM classifier object
svm_classifier = SVMClassifier()

# Train the SVM model
svm_classifier.train(X_train, y_train)

# Make predictions
X_new, y_new = da.SyntheticData(1)
predicted_label = svm_classifier.predict(X_new)
print("real:", y_new)
print("prediction:", predicted_label)
