import data as da
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class RandomForestModel:
    def __init__(self, data):
        self.data = data
        self.train_random_forest()

    def train_random_forest(self, test_size=0.2, random_state=42):
        X = self.data[['Browsing_History', 'Time_Spent']]
        y = self.data['Purchase']


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        self.model = RandomForestClassifier(random_state=random_state)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Random Forest Model Accuracy: {accuracy:.2f}')

        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))

        print('\nConfusion Matrix:')
        print(confusion_matrix(y_test, y_pred))

    def predict(self, new_data):
        return self.model.predict(new_data)

data = da.SyntheticData()

# Usage
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
