# main.py

import time
import pandas as pd
import logging
from live_predictor import LivePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/prediction.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)


def simulate_live_feed(df, predictor, delay=0, predictions=None):
    """
    Simulate a live data feed by processing the provided DataFrame sequentially.

    Args:
        df (pd.DataFrame): The test data DataFrame.
        predictor (LivePredictor): Instance of LivePredictor class.
        delay (int): Delay in seconds between processing each candle (0 for no delay).
        predictions (list): List to store prediction results.
    """
    # Ensure 'date' column is datetime and set as index
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y %H:%M', errors='coerce')
    df = df.dropna(subset=['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    print(f"Number of rows in test_data.csv: {len(df)}")

    # Pre-fill buffer with initial data
    initial_buffer = df.iloc[:predictor.buffer_size]
    for idx, row in initial_buffer.iterrows():
        new_candle = pd.DataFrame([row], index=[idx])  # Preserve datetime index
        predictor.update_buffer(new_candle)
    logging.info("Pre-filled the buffer with initial data.")

    # Start iterating from the buffer size
    for idx, row in df.iloc[predictor.buffer_size:].iterrows():
        # Create a DataFrame for the new candle with datetime index
        new_candle = pd.DataFrame([row], index=[idx])

        # Update the buffer with the new candle
        predictor.update_buffer(new_candle)

        # Calculate technical indicators
        predictor.calculate_indicators()

        # Make a prediction
        next_datetime, predicted_close = predictor.predict()

        if next_datetime and predicted_close:
            print(f"Predicted close price for {next_datetime}: {predicted_close}")
            logging.info(f"Predicted close price for {next_datetime}: {predicted_close}")
            if predictions is not None:
                predictions.append({
                    'date': next_datetime,
                    'predicted_close': predicted_close
                })

        # Wait for the specified delay to simulate real-time feed
        if delay > 0:
            time.sleep(delay)



def main():
    # Initialize the predictor
    model_path = "../data/output/model/best_model.keras"
    feature_metadata_path = "../data/output/model/selected_features.json"
    scaler_X_path = "../data/output/model/scaler_X.pkl"
    scaler_y_path = "../data/output/model/scaler_y.pkl"
    buffer_size = 50  # Adjust based on your indicator requirements
    time_steps = 10  # Must match the model's training time_steps

    # Initialize the LivePredictor with both scalers
    predictor = LivePredictor(
        model_path=model_path,
        feature_metadata_path=feature_metadata_path,
        scaler_X_path=scaler_X_path,
        scaler_y_path=scaler_y_path,
        buffer_size=buffer_size,
        time_steps=time_steps
    )

    # Initialize an empty list to store predictions
    predictions = []

    # Path to the test data CSV
    test_data_path = "../data/input/test_data.csv"

    # Read the test data into a DataFrame
    df_test = pd.read_csv(test_data_path)
    print(f"Number of rows in test_data.csv: {len(df_test)}")  # Corrected

    # Optional: Set delay in seconds (e.g., 5 minutes = 300 seconds)
    # For faster simulation, set delay=0
    delay = 0  # Change to 300 for real-time simulation

    logging.info("Starting live data feed simulation...")
    simulate_live_feed(df_test, predictor, delay, predictions)
    logging.info("Live data feed simulation completed.")

    # After simulation, create the new CSV with predictions
    create_predictions_csv(df_test, predictions)


def create_predictions_csv(df_test, predictions, output_path="../data/output/predictions_with_close.csv"):
    """
    Merge predictions with the test data and save to a new CSV file.

    Args:
        df_test (pd.DataFrame): Original test data.
        predictions (list): List of prediction dictionaries.
        output_path (str): Path to save the new CSV file.
    """
    if not predictions:
        logging.warning("No predictions to merge. Exiting CSV creation.")
        return

    # Convert predictions list to DataFrame
    df_predictions = pd.DataFrame(predictions)

    # Ensure 'date' columns are datetime for accurate merging
    df_predictions['date'] = pd.to_datetime(df_predictions['date'])
    df_test['date'] = pd.to_datetime(df_test['date'], format='%d/%m/%y %H:%M', errors='coerce')

    # Sort both DataFrames by date
    df_predictions.sort_values('date', inplace=True)
    df_test.sort_values('date', inplace=True)

    # Since predictions are for the next timestamp, shift the 'date' in predictions back by 5 minutes
    df_predictions['date'] = df_predictions['date'] - pd.Timedelta(minutes=5)

    # Merge on 'date'
    df_merged = pd.merge(df_test, df_predictions, how='left', on='date')

    # Calculate the percentage matching value
    df_merged['% matching value'] = ((df_merged['predicted_close'] - df_merged['close']) / df_merged['close']) * 100

    # Optional: Round the percentage to two decimal places
    df_merged['% matching value'] = df_merged['% matching value'].round(2)

    # Save to a new CSV file
    df_merged.to_csv(output_path, index=False)
    logging.info(f"Predictions merged and saved to {output_path}")
    print(f"Predictions merged and saved to {output_path}")


if __name__ == "__main__":
    main()
