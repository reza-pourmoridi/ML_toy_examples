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

    def predict(self, new_data):
        return self.model.predict(new_data)




data = da.SyntheticData()
random_forest = RandomForestModel(data)

new_data = da.predictionData(10)

prediction = random_forest.predict(new_data)

print(new_data)
print('prediction is')
print(prediction)