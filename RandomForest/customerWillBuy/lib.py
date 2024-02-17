# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class OnlinePurchasePredictor:
    def __init__(self, data, target_column, features_columns):
        self.data = data
        self.target_column = target_column
        self.features_columns = features_columns
        self.model = RandomForestClassifier()

    def preprocess_data(self):
        X = self.data[self.features_columns]
        y = self.data[self.target_column]
        return X, y

    def train_model(self, test_size=0.2, random_state=42):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def predict(self, new_data):
        return self.model.predict(new_data)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

# Example usage:
if __name__ == "__main__":
    # Assuming you have a DataFrame named 'df' with columns 'time_spent', 'page_views', and 'purchase' (target)
    df = ...  # Load your data here

    # Define the columns
    target_column = 'purchase'
    features_columns = ['time_spent', 'page_views']

    # Create an instance of the OnlinePurchasePredictor class
    predictor = OnlinePurchasePredictor(data=df, target_column=target_column, features_columns=features_columns)

    # Train the model and get the test set
    X_test, y_test = predictor.train_model()

    # Evaluate the model
    accuracy, report = predictor.evaluate_model(X_test, y_test)

    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:\n", report)
