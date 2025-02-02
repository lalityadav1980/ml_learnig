import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.python.client import device_lib

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Calculate Technical Indicators
def calculate_indicators(df):
    logging.info("Calculating technical indicators...")
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df['ATR'] = calculate_atr(df)
    df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['close'])
    df['Momentum'] = df['close'] - df['close'].shift(10)
    df['ROC'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
    df['High_Low_Spread'] = df['high'] - df['low']
    df['Open_Close_Spread'] = df['open'] - df['close']
    return df.dropna()

# RSI Calculation
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ATR Calculation
def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

# Bollinger Bands Calculation
def calculate_bollinger_bands(series, window=20, num_sd=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (num_sd * std)
    lower_band = sma - (num_sd * std)
    return upper_band, lower_band

# Prepare LSTM Input Data
def prepare_lstm_data(X, y, time_steps):
    logging.info("Preparing LSTM input data...")
    X_lstm, y_lstm = [], []
    for i in range(len(X) - time_steps):
        X_lstm.append(X[i:i + time_steps])
        y_lstm.append(y[i + time_steps])
    return np.array(X_lstm), np.array(y_lstm)

# Plot Training Loss
def plot_training_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot Predictions vs Actual
def plot_predictions(y_actual, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main Program
def main():
    # List all available devices
    print(device_lib.list_local_devices())
    # Check if TensorFlow sees a GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    logging.info("Loading dataset...")
    df = pd.read_csv("../data/NIFTY_5_minute.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    logging.info("Performing feature engineering...")
    df = calculate_indicators(df)

    logging.info("Defining features and target...")
    X = df.drop(columns=['close'])
    y = df['close']

    logging.info("Performing feature selection...")
    rf = RandomForestRegressor()
    rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_squared_error')
    rfecv.fit(X, y)
    X_selected = X.iloc[:, rfecv.support_]

    logging.info("Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, shuffle=False)

    logging.info("Scaling data...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare Data for LSTM
    time_steps = 10
    X_train_lstm, y_train_lstm = prepare_lstm_data(X_train_scaled, y_train.values, time_steps)
    X_test_lstm, y_test_lstm = prepare_lstm_data(X_test_scaled, y_test.values, time_steps)

    logging.info("Building LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    logging.info("Training LSTM model...")
    history = model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test_lstm), verbose=1)

    # Plot Training Loss
    plot_training_loss(history)

    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test_lstm)
    y_test_actual = y_test_lstm
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Metrics
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    logging.info(f"Mean Absolute Error (MAE): {mae}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot Predictions
    plot_predictions(y_test_actual, y_pred_actual)

if __name__ == "__main__":
    main()
