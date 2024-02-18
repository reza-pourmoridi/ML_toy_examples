import numpy as np
import pandas as pd

# Set random seed for reproducibility
def SyntheticData(num_samples = 1000):
    np.random.seed(42)
    # Generate synthetic data
    browsing_history = np.random.randint(1, 11, size=num_samples)  # Number of pages visited
    time_spent = np.random.uniform(1, 60, size=num_samples)  # Time spent on the website in minutes

    # Assuming a simple pattern where more browsing and longer time increase the likelihood of purchase
    # You can adjust these coefficients based on the desired pattern
    purchase_likelihood = 0.3 * browsing_history + 0.5 * time_spent + np.random.normal(0, 5, size=num_samples)

    # Create a DataFrame
    data = pd.DataFrame({
        'Browsing_History': browsing_history,
        'Time_Spent': time_spent,
        'Purchase_Likelihood': purchase_likelihood
    })

    # Add a binary column indicating whether the customer made a purchase (1) or not (0)
    data['Purchase'] = (data['Purchase_Likelihood'] > data['Purchase_Likelihood'].median()).astype(int)

    # Remove the 'Purchase_Likelihood' column
    data = data.drop('Purchase_Likelihood', axis=1)
    return data

def predictionData(num_samples = 1):
    new_data = pd.DataFrame({
        'Browsing_History': np.random.randint(1, 11, size=num_samples),
        'Time_Spent': np.random.uniform(1, 60, size=num_samples)
    })
    return new_data
