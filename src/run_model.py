import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from train_lstm import calculate_indicators, calculate_rsi, calculate_atr, calculate_bollinger_bands, \
    check_data_integrity
import os
import json

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Load the calculate_indicators and check_data_integrity functions
# Assuming these are already defined in your script as shown earlier
def predict_next_close(model_path, test_file, feature_metadata_path, time_steps=10):
    """
    Predict the next close price iteratively, starting with the first `time_steps` candles.

    Args:
        model_path (str): Path to the pre-trained model.
        test_file (str): Path to the test data CSV file.
        feature_metadata_path (str): Path to the JSON file containing selected features.
        time_steps (int): Number of time steps to use for prediction.

    Returns:
        list: List of predictions as tuples (datetime, predicted_close_price).
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load the model
    logging.info(f"Loading pre-trained model from {model_path}")
    model = load_model(model_path)

    # Load selected features
    if not os.path.exists(feature_metadata_path):
        raise FileNotFoundError(f"Feature metadata not found at {feature_metadata_path}")
    with open(feature_metadata_path, 'r') as f:
        selected_features = json.load(f)
    logging.info(f"Loaded selected features: {selected_features}")

    # Load the test dataset
    logging.info(f"Loading test dataset from {test_file}")
    df = pd.read_csv(test_file)

    # Convert date column to datetime and set it as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Initialize predictions list
    predictions = []

    # Iterate over windows of data for predictions
    for i in range(time_steps, len(df) + 1):
        # Slice the data for the current prediction window
        current_window = df.iloc[i - time_steps:i].copy()

        # Check if the current window has enough rows
        if len(current_window) < time_steps:
            logging.warning(f"Insufficient data for prediction at iteration {i}. Skipping...")
            continue

        # Calculate technical indicators for the current window
        logging.info(f"Calculating technical indicators for prediction window ending at {current_window.index[-1]}")
        current_window = calculate_indicators(current_window)

        # Ensure only selected features are used
        try:
            current_window = current_window[selected_features + ['close']]
        except KeyError as e:
            logging.error(f"Missing selected features in the data. Error: {e}")
            continue

        # Skip if there are NaN values or insufficient rows after processing
        if current_window.isna().any().any() or len(current_window) < time_steps:
            logging.warning(f"Insufficient data after processing indicators at iteration {i}. Skipping...")
            continue

        # Scale the data
        logging.info("Scaling data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(current_window[selected_features + ['close']])
        X_input = scaled_data[:, :-1]  # All features except 'close'

        # Prepare LSTM input data for prediction
        X_input = np.expand_dims(X_input, axis=0)  # Add batch dimension

        # Predict the next close price
        logging.info("Predicting the next close price...")
        predicted_scaled_close = model.predict(X_input)[0][0]

        # Reverse the scaling for close price
        scaler_for_close = StandardScaler()
        scaler_for_close.fit(current_window[['close']])  # Fit only on the 'close' column
        predicted_close = scaler_for_close.inverse_transform([[predicted_scaled_close]])[0][0]

        # Determine the datetime of the next prediction
        next_datetime = current_window.index[-1] + pd.Timedelta(minutes=5)
        logging.info(f"Predicted close price for {next_datetime}: {predicted_close}")

        # Append prediction to the results
        predictions.append((next_datetime, predicted_close))

    # Return the list of predictions
    if not predictions:
        logging.error("No valid predictions were generated.")
        return []

    return predictions


# Run the prediction
if __name__ == "__main__":
    best_model_path = "../data/output/model/best_model.keras"
    test_data_path = "../data/input/test_data.csv"
    feature_metadata_path = "../data/output/model/selected_features.json"

    predicted_close, next_datetime = predict_next_close(best_model_path, test_data_path, feature_metadata_path)
    print(f"Predicted close price: {predicted_close}")
    print(f"Next datetime: {next_datetime}")
