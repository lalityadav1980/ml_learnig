# live_predictor.py

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


class LivePredictor:
    def __init__(self, model_path, feature_metadata_path, scaler_X_path, scaler_y_path, buffer_size=50, time_steps=10):
        """
        Initialize the LivePredictor.

        Args:
            model_path (str): Path to the pre-trained model.
            feature_metadata_path (str): Path to the JSON file containing selected features.
            scaler_X_path (str): Path to the saved scaler for features.
            scaler_y_path (str): Path to the saved scaler for target.
            buffer_size (int): Number of data points to keep in the buffer.
            time_steps (int): Number of time steps for the LSTM model.
        """
        self.model_path = model_path
        self.feature_metadata_path = feature_metadata_path
        self.scaler_X_path = scaler_X_path
        self.scaler_y_path = scaler_y_path
        self.buffer_size = buffer_size
        self.time_steps = time_steps

        # Load the model
        self.model = self.load_model()

        # Load selected features
        self.selected_features = self.load_features()

        # Initialize buffer
        self.buffer = pd.DataFrame()

        # Initialize scalers
        self.scaler_X, self.scaler_y = self.load_scalers()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        logging.info(f"Loading pre-trained model from {self.model_path}")
        return load_model(self.model_path)

    def load_features(self):
        if not os.path.exists(self.feature_metadata_path):
            raise FileNotFoundError(f"Feature metadata not found at {self.feature_metadata_path}")
        with open(self.feature_metadata_path, 'r') as f:
            selected_features = json.load(f)
        logging.info(f"Loaded selected features: {selected_features}")
        return selected_features

    def load_scalers(self):
        if not os.path.exists(self.scaler_X_path):
            raise FileNotFoundError(f"Features scaler not found at {self.scaler_X_path}")
        if not os.path.exists(self.scaler_y_path):
            raise FileNotFoundError(f"Target scaler not found at {self.scaler_y_path}")
        logging.info(f"Loading feature scaler from {self.scaler_X_path}")
        with open(self.scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        logging.info(f"Loading target scaler from {self.scaler_y_path}")
        with open(self.scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)

        # Correct logging using local variables
        logging.info(f"Feature scaler mean: {scaler_X.mean_}")
        logging.info(f"Target scaler mean: {scaler_y.mean_}")

        return scaler_X, scaler_y

    def update_buffer(self, new_data):
        """
        Update the data buffer with new incoming data.

        Args:
            new_data (pd.DataFrame): New OHLC data to append.
        """
        self.buffer = pd.concat([self.buffer, new_data])
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer.iloc[-self.buffer_size:]
        logging.info(f"Buffer updated. Current buffer size: {len(self.buffer)}")
        logging.debug(f"Updated buffer index type: {self.buffer.index.dtype}")
        logging.debug(f"Updated buffer:\n{self.buffer}")

    def calculate_indicators(self):
        """
        Calculate technical indicators on the current buffer.
        """
        df = self.buffer.copy()
        logging.info("Calculating technical indicators...")
        if len(df) < 20:
            logging.warning("Not enough data to calculate technical indicators.")
            return
        # Calculate indicators
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['RSI'] = self.calculate_rsi(df['close'])
        df['ATR'] = self.calculate_atr(df)
        df['Bollinger_Upper'], df['Bollinger_Lower'] = self.calculate_bollinger_bands(df['close'])
        df['Momentum'] = df['close'] - df['close'].shift(10)
        df['ROC'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        df['High_Low_Spread'] = df['high'] - df['low']
        df['Open_Close_Spread'] = df['open'] - df['close']
        df['RSI_Momentum'] = df['RSI'] - df['RSI'].shift(1)
        df['RSI_Momentum_Percentage'] = (df['RSI'] - df['RSI'].shift(1)) / df['RSI'].shift(1) * 100

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Only drop the latest row if it has any NaN values
        if df.iloc[-1].isna().any():
            df = df.iloc[:-1]
            logging.info("Dropped the latest row due to NaN values in indicators.")
        else:
            logging.info("All indicators calculated for the latest row.")

        self.buffer = df
        logging.info("Technical indicators calculated and buffer updated.")

    def calculate_rsi(self, series, window=14):
        """
        Calculate Relative Strength Index (RSI).

        Args:
            series (pd.Series): Series of closing prices.
            window (int): Period for RSI calculation.

        Returns:
            pd.Series: RSI values.
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, window=14):
        """
        Calculate Average True Range (ATR).

        Args:
            df (pd.DataFrame): DataFrame containing OHLC data.
            window (int): Period for ATR calculation.

        Returns:
            pd.Series: ATR values.
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

    def calculate_bollinger_bands(self, series, window=20, num_std=2):
        """
        Calculate Bollinger Bands.

        Args:
            series (pd.Series): Series of closing prices.
            window (int): Window size for moving average.
            num_std (int): Number of standard deviations for the bands.

        Returns:
            tuple: Upper and lower Bollinger Bands.
        """
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        return upper_band, lower_band

    def preprocess_data(self):
        """
        Preprocess the data: select features and scale.

        Returns:
            np.array: Preprocessed data ready for prediction.
        """
        # Ensure buffer has enough data
        if len(self.buffer) < self.time_steps:
            logging.warning("Not enough data to make a prediction.")
            return None

        # Select the last 'time_steps' data points
        window = self.buffer.iloc[-self.time_steps:].copy()

        # Select necessary features
        try:
            window = window[self.selected_features]
        except KeyError as e:
            logging.error(f"Missing selected features in the data. Error: {e}")
            return None

        # Handle NaN values
        if window.isna().any().any():
            logging.warning("NaN values found in the window. Skipping prediction.")
            return None

        # Scale the data using the pre-fitted scaler_X
        scaled_data = self.scaler_X.transform(window)
        X_input = scaled_data  # Already scaled
        X_input = np.expand_dims(X_input, axis=0)  # Add batch dimension
        logging.debug(f"Preprocessed data shape: {X_input.shape}")
        return X_input

    def predict(self):
        """
        Make a prediction for the next close price.

        Returns:
            tuple: (next_datetime, predicted_close)
        """
        # Preprocess data
        X_input = self.preprocess_data()
        if X_input is None:
            return None, None

        # Predict
        predicted_scaled_close = self.model.predict(X_input)[0][0]

        # Reverse the scaling for 'close'
        # Since y was scaled separately, use scaler_y to inverse transform
        predicted_close = self.scaler_y.inverse_transform([[predicted_scaled_close]])[0][0]

        # Determine the datetime for the next prediction
        last_datetime = self.buffer.index[-1]
        next_datetime = last_datetime + pd.Timedelta(minutes=5)

        logging.info(f"Predicted close price for {next_datetime}: {predicted_close}")
        return next_datetime, predicted_close
