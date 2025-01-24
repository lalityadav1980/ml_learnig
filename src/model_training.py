from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

def load_data():
    from sklearn.datasets import fetch_openml
    try:
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        data = boston.frame
        data.columns = [col.lower() for col in data.columns]  # Ensure column names are lowercase
        data.rename(columns={'medv': 'price'}, inplace=True)  # Rename target column to 'price'
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def train_model(data):
    X = data.drop('price', axis=1)  # Assuming 'price' is the target variable
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    # Load dataset
    data = load_data()
    if data is not None:
        # Train model
        model, X_test, y_test = train_model(data)
        # Save the trained model
        save_model(model, 'linear_regression_model.pkl')