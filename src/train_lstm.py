import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers  # For L1/L2 regularization
import tensorflow as tf  # Ensure TensorFlow is imported
import logging
import json
import math
import os
import pickle  # For saving and loading scalers
import talib

from src.indicator import supertrend

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def calculate_indicators(df):
    logging.info("Starting calculation of technical indicators using TA-Lib...")

    # Calculate ADX
    logging.info("Calculating ADX")
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    # Drop rows where ADX is NaN
    df = df.dropna(subset=['ADX'])
    logging.debug("ADX calculated. Current columns: %s", df.columns.tolist())

    # Calculate SuperTrend indicator
    logging.info("Calculating SuperTrend indicator")
    df = supertrend(df, period=10, multiplier=3)  # Your SuperTrend function should add a column 'STX'
    logging.debug("SuperTrend calculated. Current columns: %s", df.columns.tolist())

    # Replace infinite values with NaN and drop rows that contain any NaN values
    logging.info("Replacing infinite values and dropping NaNs")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    logging.debug("After replace, DataFrame shape: %s", df.shape)
    df.dropna(inplace=True)
    logging.debug("After dropna, DataFrame shape: %s", df.shape)

    logging.info("Completed calculating technical indicators. Final columns: %s", df.columns.tolist())
    print(df.tail())
    return df


def supertrend_strategy(df, atr_multiplier=1.5, adx_threshold=20):
    """
    Build a trading strategy based on the SuperTrend indicator filtered by ADX,
    and calculate a dynamic stop-loss based on a multiple of ATR.

    Parameters:
      - df: DataFrame with at least 'open', 'high', 'low', 'close' columns.
      - atr_multiplier: The multiple of ATR to use for the stop-loss.
      - adx_threshold: The ADX threshold (e.g., 20) above which we consider the trend strong.

    Returns:
      The DataFrame with added columns:
        - 'ATR': Average True Range.
        - 'ADX': Average Directional Index.
        - 'STX': SuperTrend direction ('up' or 'down') computed by SuperTrend().
        - 'supertrend_signal': 'Buy', 'Sell', or 'NoTrade'.
        - 'stop_loss': Calculated stop-loss level based on the ATR.
    """
    # Calculate ATR and ADX using TA-Lib (14-period)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # Compute SuperTrend indicator; this should add a column 'STX'
    df = supertrend(df, period=10, multiplier=3)  # 'STX' should be 'up' or 'down'

    # Generate strategy signal using SuperTrend filtered by ADX.
    df['supertrend_signal'] = np.where(
        (df['ADX'] > adx_threshold) & (df['STX'] == 'up'),
        'Buy',
        np.where(
            (df['ADX'] > adx_threshold) & (df['STX'] == 'down'),
            'Sell',
            'NoTrade'
        )
    )

    # Calculate stop-loss based on ATR:
    # For Buy signal: stop-loss = close - (atr_multiplier * ATR)
    # For Sell signal: stop-loss = close + (atr_multiplier * ATR)
    df['stop_loss'] = np.where(
        df['supertrend_signal'] == 'Buy',
        df['close'] - (atr_multiplier * df['ATR']),
        np.where(
            df['supertrend_signal'] == 'Sell',
            df['close'] + (atr_multiplier * df['ATR']),
            np.nan  # No stop loss for 'NoTrade' signals.
        )
    )

    return df


def generate_strategy_signals(df):
    """
    Generate strategy signals by combining multiple indicators, including the SuperTrend
    strategy with ATR, ADX, RSI, Bollinger Bands, EMA, MACD, and Stochastic Oscillator.

    After signal generation, drop intermediate columns that are not needed as final features.
    """
    # Generate signals from the SuperTrend strategy
    df = supertrend_strategy(df, atr_multiplier=1.5, adx_threshold=20)
    df = df.dropna(subset=['ADX'])
    df = calculate_profit(df)

    # With max-absolute scaling:
    df = assign_continuous_rewards(df,
                                   penalty_multiplier=2.0,
                                   min_profit_threshold=50,
                                   small_profit_penalty=20,
                                   scale_rewards=True,
                                   scaling_method="maxabs")
    print("\nMaxAbs-scaled rewards:\n", df.head())

    return df

def assign_continuous_rewards(df,
                              penalty_multiplier=2.0,
                              min_profit_threshold=50,
                              small_profit_penalty=20,
                              scale_rewards=False,
                              scaling_method="maxabs"):
    """
    Compute a continuous reward from the 'profit' column, with extra penalty for trades
    whose profit is below a specified threshold.

    Rules:
      - If profit >= min_profit_threshold:
            reward = profit
      - If 0 < profit < min_profit_threshold:
            reward = profit - small_profit_penalty
      - If profit < 0:
            reward = profit * penalty_multiplier

    Optionally scales the rewards according to the chosen scaling_method.

    Parameters:
      df (DataFrame): Must have a numeric 'profit' column.
      penalty_multiplier (float): Multiplier for negative profits.
      min_profit_threshold (float): Minimum profit to be considered fully rewarding.
      small_profit_penalty (float): Fixed penalty to subtract if profit is positive but less than threshold.
      scale_rewards (bool): If True, scale rewards.
      scaling_method (str): One of "maxabs", "minmax", or "zscore".

    Returns:
      DataFrame: A copy of df with a new column 'reward'.
    """
    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Create a vectorized reward calculation.
    # For profit >= min_profit_threshold, reward = profit.
    # For 0 < profit < min_profit_threshold, reward = profit - small_profit_penalty.
    # For profit <= 0, reward = profit * penalty_multiplier.
    profit = df['profit'].values  # get NumPy array for speed
    reward = np.where(profit >= min_profit_threshold,
                      profit,
                      np.where(profit > 0,
                               profit - small_profit_penalty,
                               profit * penalty_multiplier))

    df['reward'] = reward

    # Optionally scale rewards using vectorized operations.
    if scale_rewards:
        if scaling_method == "maxabs":
            max_abs = np.abs(df['reward']).max()
            if max_abs != 0:
                df['reward'] = df['reward'] / max_abs
        elif scaling_method == "minmax":
            min_val = df['reward'].min()
            max_val = df['reward'].max()
            if max_val - min_val != 0:
                df['reward'] = (df['reward'] - min_val) / (max_val - min_val)
        elif scaling_method == "zscore":
            mean_val = df['reward'].mean()
            std_val = df['reward'].std()
            if std_val != 0:
                df['reward'] = (df['reward'] - mean_val) / std_val
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

    return df


def calculate_profit(df):
    """
    Calculate profit for each row based on changes in the 'supertrend_signal'
    column, using the following rules:

      1. If (current_signal == 'Buy' and previous_signal == 'Buy') or
         (current_signal == 'Sell' and previous_signal == 'Sell'):
             → The trade is continuing; profit = 0.

      2. If (current_signal == 'Buy' and previous_signal == 'Sell') or
         (current_signal == 'Sell' and previous_signal == 'Buy'):
             → This is a reversal. Calculate profit on the previous trade (exit at current_close),
                then open a new trade with the new signal at current_close.
             For a previous Buy trade: profit = current_close – entry_price.
             For a previous Sell trade: profit = entry_price – current_close.

      3. If (current_signal == 'Buy' and previous_signal == 'NoTrade') or
         (current_signal == 'Sell' and previous_signal == 'NoTrade'):
             → This marks a trade entry. Capture entry_price = current_close; profit = 0.

      4. If current_signal == 'NoTrade' and previous_signal in ['Buy', 'Sell']:
             → This marks a trade exit. For a Buy trade, profit = current_close – entry_price;
                for a Sell trade, profit = entry_price – current_close.
             After computing profit, close the trade.

      5. If current_signal == 'NoTrade' and previous_signal == 'NoTrade':
             → No trade is active; profit = 0.

    The function adds a 'profit' column to the DataFrame.

    Note: It assumes the DataFrame is already sorted in chronological order.
    """
    # Work on a copy and reset the index for integer-based indexing.
    df = df.copy().reset_index(drop=True)
    profits = [0.0] * len(df)

    # State variables to keep track of an open trade
    trade_open = False
    trade_type = None  # Will be either 'Buy' or 'Sell'
    entry_price = None

    # Assume that before the first row the signal was 'NoTrade'
    prev_signal = 'NoTrade'

    # Loop over the DataFrame rows.
    for i in range(len(df)):
        current_signal = df.loc[i, 'supertrend_signal']
        current_close = df.loc[i, 'close']

        # Condition 1: Signal continues (Buy→Buy or Sell→Sell)
        if (current_signal == 'Buy' and prev_signal == 'Buy') or \
                (current_signal == 'Sell' and prev_signal == 'Sell'):
            profits[i] = 0.0
            # State remains unchanged.

        # Condition 2: Signal reversal (Buy after Sell or Sell after Buy)
        elif (current_signal == 'Buy' and prev_signal == 'Sell') or \
                (current_signal == 'Sell' and prev_signal == 'Buy'):
            if trade_open:
                # Close previous trade and compute profit.
                if trade_type == 'Buy':
                    profit_value = current_close - entry_price
                elif trade_type == 'Sell':
                    profit_value = entry_price - current_close
                profits[i] = profit_value
                # Then immediately open a new trade with the new signal.
                trade_type = current_signal
                entry_price = current_close
                # trade_open remains True.
            else:
                # If no trade was open, simply start a trade.
                trade_open = True
                trade_type = current_signal
                entry_price = current_close
                profits[i] = 0.0

        # Condition 3: Trade entry (Buy or Sell initiated from NoTrade)
        elif (current_signal == 'Buy' and prev_signal == 'NoTrade') or \
                (current_signal == 'Sell' and prev_signal == 'NoTrade'):
            trade_open = True
            trade_type = current_signal
            entry_price = current_close
            profits[i] = 0.0

        # Condition 4: Trade exit (transition from an active trade to NoTrade)
        elif current_signal == 'NoTrade' and prev_signal in ['Buy', 'Sell']:
            if trade_open:
                if trade_type == 'Buy':
                    profit_value = current_close - entry_price
                elif trade_type == 'Sell':
                    profit_value = entry_price - current_close
                profits[i] = profit_value
                # Close the trade.
                trade_open = False
                trade_type = None
                entry_price = None
            else:
                profits[i] = 0.0

        # Condition 5: No trade continues (NoTrade → NoTrade)
        elif current_signal == 'NoTrade' and prev_signal == 'NoTrade':
            profits[i] = 0.0

        else:
            # Catch-all (should not occur if signals are one of the three allowed values).
            profits[i] = 0.0

        # Update prev_signal for the next iteration.
        prev_signal = current_signal

    df['profit'] = profits
    return df


def prepare_lstm_data(X, y, dates, time_steps):
    logging.info("Preparing LSTM input data...")
    X_lstm, y_lstm, y_dates = [], [], []
    for i in range(len(X) - time_steps):
        X_lstm.append(X[i:i + time_steps])
        y_lstm.append(y[i + time_steps])
        y_dates.append(dates[i + time_steps])
    return np.array(X_lstm), np.array(y_lstm), np.array(y_dates)


def split_data_with_dates(X, y, dates, train_ratio=0.8):
    logging.info("Splitting the data into training, validation, and testing sets with dates...")
    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = int(train_end + (total_samples - train_end) / 2)

    X_train = X[:train_end]
    y_train = y[:train_end]
    dates_train = dates[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    dates_val = dates[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]
    dates_test = dates[val_end:]

    logging.info(f"Data split into train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test


def build_lstm_model(input_shape):
    logging.info("Building the LSTM model with regularization and gradient clipping...")
    model = Sequential([
        Input(shape=input_shape),
        # L1/L2 Regularization added to LSTM layers
        LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.3),  # Dropout to prevent overfitting
        LSTM(64, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.3),
        Dense(1)
    ])
    # Adam optimizer with gradient clipping
    optimizer = Adam(learning_rate=0.0001, clipnorm=5.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['RootMeanSquaredError'])
    return model


def scheduler(epoch, lr):
    return max(lr * 0.9, 5e-5)


def get_callbacks():
    checkpoint = ModelCheckpoint('../data/output/model/best_model.keras',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=20,
                                   restore_best_weights=True,
                                   verbose=1)
    lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
    detailed_loss_logger = DetailedLossLogger()
    return [checkpoint, early_stopping, lr_scheduler, detailed_loss_logger]


class DetailedLossLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logging.info(
            f"Epoch {epoch + 1}: "
            f"Training Loss = {logs.get('loss', 0):.10f}, "
            f"Validation Loss = {logs.get('val_loss', 0):.10f}"
        )


def clear_old_charts(directory="../data/output/charts"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for file in os.listdir(directory):
            if file.endswith(".png"):
                os.remove(os.path.join(directory, file))
    logging.info(f"Cleared old charts in the directory: {directory}")


def plot_training_and_validation_loss(history, save_dir="../data/output/charts"):
    """
    Plot training and validation loss over epochs to visualize overfitting.

    Parameters:
    - history (History): Keras History object returned by model.fit().
    - save_dir (str): Directory to save the chart.
    """
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logging.info(f"Created directory for charts at: {save_dir}")

    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Training Loss', color='green')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the chart
    chart_path = os.path.join(save_dir, "training_validation_loss.png")
    plt.savefig(chart_path)
    logging.info(f"Saved training vs validation loss chart to: {chart_path}")
    plt.close()


def plot_predictions(y_actual, y_pred, save_dir="../data/output/charts", sample_size=300):
    """
    Plot actual vs predicted values to visualize model performance.

    Parameters:
    - y_actual (np.array): Actual target values (inverse-transformed if scaled).
    - y_pred (np.array): Predicted target values (inverse-transformed if scaled).
    - save_dir (str): Directory where the chart will be saved.
    - sample_size (int): Number of latest samples to plot for better visualization.
    """
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logging.info(f"Created directory for charts at: {save_dir}")

    # Reduce data for plotting if sample_size is specified
    if sample_size > 0 and sample_size < len(y_actual):
        y_actual = y_actual[-sample_size:]
        y_pred = y_pred[-sample_size:]

    plt.figure(figsize=(14, 7))
    plt.plot(y_actual, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red')
    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the chart
    chart_path = os.path.join(save_dir, "actual_vs_predicted_prices.png")
    plt.savefig(chart_path)
    logging.info(f"Saved actual vs predicted prices chart to: {chart_path}")
    plt.close()


def plot_predictions_full(y_actual, y_pred, save_dir="../data/output/charts"):
    """
    Plot all actual vs predicted values without reducing the sample size.

    Parameters:
    - y_actual (np.array): Actual target values (inverse-transformed if scaled).
    - y_pred (np.array): Predicted target values (inverse-transformed if scaled).
    - save_dir (str): Directory where the chart will be saved.
    """
    plot_predictions(y_actual, y_pred, save_dir=save_dir, sample_size=len(y_actual))


def add_noise(data, noise_level=0.01):
    """
    Add Gaussian noise to the data for data augmentation.

    Parameters:
    - data: Numpy array of data.
    - noise_level: Standard deviation of the Gaussian noise.

    Returns:
    - Noisy data as a numpy array.
    """
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise


def save_to_csv(data, filename):
    if isinstance(data, pd.DataFrame):
        data.to_csv(filename, index=False)
    elif isinstance(data, pd.Series):
        data.to_csv(filename, header=True)


def get_feature_correlations(df, target='close', output_file="../data/output/data/correlations.csv"):
    correlation_matrix = df.corr()
    correlations = correlation_matrix[target].drop(target)
    sorted_correlations = correlations.abs().sort_values(ascending=False)
    save_to_csv(sorted_correlations, output_file)
    logging.info(f"Saved correlations to {output_file}")
    return sorted_correlations


def cross_validate_features(features, df, model, output_file="../output/data/cross_validation_results.csv"):
    X = df[features]
    y = df['close']
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_rmse = np.sqrt(-scores.mean())
    pd.DataFrame({"Feature Set": [features], "Mean RMSE": [mean_rmse]}).to_csv(output_file, index=False)
    logging.info(f"Saved cross-validation results to {output_file}")
    return mean_rmse


def check_data_integrity(df):
    """
    Check the DataFrame for NaN, inf, or very large values.
    """
    # Check for NaN values
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        logging.error(f"Columns with NaN values: {nan_cols}")
        raise ValueError("Data contains NaN values. Please handle them before proceeding.")

    # Check for infinite values
    if not np.isfinite(df.to_numpy()).all():
        inf_cols = df.columns[(~np.isfinite(df)).any()].tolist()
        logging.error(f"Columns with infinite values: {inf_cols}")
        raise ValueError("Data contains infinite values. Please handle them before proceeding.")

    logging.info("Data integrity check passed.")


def measure_best_features(df, features, output_dir="../data/output/csv", correlation_threshold=0.1):
    """
    Measure feature correlations and log which features are eliminated and which are used.

    Args:
        df (pd.DataFrame): The DataFrame containing all features and the target.
        features (list): The list of candidate features to evaluate.
        output_dir (str): Directory to save the results.
        correlation_threshold (float): Minimum correlation value to keep a feature.

    Returns:
        dict: A dictionary containing correlations, eliminated features, and selected features.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Feature correlation analysis
    logging.info("Analyzing feature correlations...")
    correlation_file = os.path.join(output_dir, "feature_correlations.csv")
    correlations = df[features + ['close']].corr()['close'].drop('close')  # Correlation with the target 'close'
    save_to_csv(correlations, correlation_file)
    logging.info(f"Saved all feature correlations to {correlation_file}")

    # Step 2: Identify low-correlation features
    low_correlation_features = correlations[correlations.abs() < correlation_threshold].index.tolist()
    logging.info(f"Features eliminated due to low correlation (<{correlation_threshold}): {low_correlation_features}")

    # Save eliminated features to a file
    eliminated_file = os.path.join(output_dir, "eliminated_features.csv")
    pd.DataFrame(low_correlation_features, columns=["Eliminated Features"]).to_csv(eliminated_file, index=False)
    logging.info(f"Saved eliminated features to {eliminated_file}")

    # Step 3: Keep only high-correlation features
    selected_features = correlations[correlations.abs() >= correlation_threshold].index.tolist()
    logging.info(f"Features selected for the final model: {selected_features}")

    # Save selected features to a file
    selected_file = os.path.join(output_dir, "selected_features.csv")
    pd.DataFrame(selected_features, columns=["Selected Features"]).to_csv(selected_file, index=False)
    logging.info(f"Saved selected features to {selected_file}")

    return {
        "correlations": correlations,
        "eliminated_features": low_correlation_features,
        "selected_features": selected_features
    }


def save_scaler(scaler, file_path):
    """
    Save the fitted scaler to a file using pickle.

    Args:
        scaler (StandardScaler): Fitted scaler object.
        file_path (str): Path to save the scaler.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {file_path}")


def save_selected_features(features, file_path):
    """
    Save selected features to a JSON file.

    Args:
        features (list): List of selected feature names.
        file_path (str): Path to save the JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(features, f)
    logging.info(f"Selected features saved to {file_path}")


def main():
    # Define paths
    best_model_path = "../data/output/model/best_model.keras"
    feature_metadata_path = "../data/output/model/selected_features.json"
    scaler_x_path = "../data/output/model/scaler_X.pkl"
    scaler_y_path = "../data/output/model/scaler_y.pkl"
    charts_dir = "../data/output/charts"
    data_dir = "../data/output"
    predictions_csv_path = "../data/output/predictions.csv"  # Path to save predictions CSV
    # Save the Fibonacci retracement levels
    output_csv_dir = '../data/output/csv'
    os.makedirs(output_csv_dir, exist_ok=True)
    indicators_csv_path = os.path.join(output_csv_dir, 'output_indicators.csv')
    strategy_csv_path = os.path.join(output_csv_dir, 'final_strategy_signals.csv')

    # Load dataset
    logging.info("Loading dataset...")
    df = pd.read_csv("../data/input/nifty_5_minute_60_days.csv")

    # Define the date format matching your data
    date_format = '%Y-%m-%d %H:%M:%S%z'

    # Convert 'date' column to datetime using the specified format
    logging.info("Converting 'date' column to datetime...")
    df['date'] = pd.to_datetime(df['date'], format=date_format, errors='coerce')

    # Check for any parsing errors
    if df['date'].isna().any():
        logging.warning("Some dates could not be parsed and are set as NaT.")
        # Optionally handle NaT values, for example by dropping them:
        df = df.dropna(subset=['date'])

    # Set 'date' as the index
    df.set_index('date', inplace=True)

    # --- Step 1: Calculate and Save Technical Indicators ---
    if not os.path.exists(indicators_csv_path):
        logging.info("Indicators file not found. Calculating technical indicators...")
        df_indicators = calculate_indicators(df)  # Your function that computes all indicators
        df_indicators.to_csv(indicators_csv_path, index=True)
        logging.info(f"Technical indicators saved to {indicators_csv_path}")
    else:
        logging.info(f"Indicators file {indicators_csv_path} already exists. Loading indicators from CSV...")
        df_indicators = pd.read_csv(indicators_csv_path)

    # --- Step 2: Generate and Save Strategy Signals ---
    if not os.path.exists(strategy_csv_path):
        logging.info("Strategy signals file not found. Generating strategy signals...")
        # Use a copy of the indicators DataFrame to generate strategy signals
        df_strategy = generate_strategy_signals(df_indicators.copy())
        df_strategy = df_strategy.reset_index().rename(columns={'index': 'date'})
        df_strategy.to_csv(strategy_csv_path, index=False)
        logging.info(f"Strategy signals saved to {strategy_csv_path}")
    else:
        logging.info(f"Strategy signals file {strategy_csv_path} already exists. Skipping strategy signal generation.")

    logging.info("Data saved to output_indicators.csv")

    # Drop unused columns, including 'open', 'high', and 'low'
    df.drop(columns=['open', 'high', 'low', 'volume'], inplace=True, errors='ignore')

    # Perform data integrity check
    logging.info("Performing data integrity check...")
    check_data_integrity(df)

    # Feature analysis and selection
    feature_analysis = measure_best_features(df, features=[
        'SMA_10', 'EMA_10', 'RSI', 'RSI_Momentum', 'RSI_Momentum_Percentage',
        'ATR', 'Bollinger_Upper', 'Bollinger_Lower', 'Momentum', 'ROC',
        'Log_Returns', 'High_Low_Spread', 'Open_Close_Spread'
    ], output_dir=data_dir)
    selected_features = feature_analysis["selected_features"]

    # Save selected features to a JSON file
    save_selected_features(selected_features, feature_metadata_path)

    # Filter DataFrame to keep only selected features
    df = df[selected_features + ['close']]

    # Scale features and target separately
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(df[selected_features])

    scaler_y = StandardScaler()
    y = df['close'].values.reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y)

    # Save the scalers
    save_scaler(scaler_X, scaler_x_path)
    save_scaler(scaler_y, scaler_y_path)

    # Extract dates after prepare_lstm_data
    dates = df.index

    # Prepare LSTM input data
    X_lstm, y_lstm, y_dates = prepare_lstm_data(X_scaled, y_scaled, dates, time_steps=10)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = split_data_with_dates(
        X_lstm, y_lstm, y_dates, train_ratio=0.8
    )

    # Data Augmentation: Add Gaussian noise to training and validation data
    logging.info("Adding noise to training and validation data for augmentation...")
    X_train_noisy = add_noise(X_train, noise_level=0.01)
    X_val_noisy = add_noise(X_val, noise_level=0.01)

    # Check if a pre-trained model exists
    if os.path.exists(best_model_path):
        logging.info(f"Best model found at {best_model_path}. Skipping training...")
        model = load_model(best_model_path)
    else:
        logging.info("No pre-trained model found. Proceeding with training...")
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        callbacks = get_callbacks()

        # Train the model with augmented data
        history = model.fit(
            X_train_noisy, y_train,
            validation_data=(X_val_noisy, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Save the trained model
        model.save(best_model_path)
        logging.info(f"Saved trained model to {best_model_path}")

        # Save training and validation loss plots
        plot_training_and_validation_loss(history, save_dir=charts_dir)

    # Evaluate the model
    logging.info("Evaluating the model on the test set...")
    loss, rmse = model.evaluate(X_test, y_test, verbose=1)
    logging.info(f"Test Loss: {loss}, RMSE: {rmse}")

    # Process predictions and save results
    logging.info("Processing predictions and calculating RMSE on actual scale...")
    predicted_prices_scaled = model.predict(X_test)
    predicted_prices = scaler_y.inverse_transform(predicted_prices_scaled)
    actual_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    rmse_actual = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
    logging.info(f"RMSE (Actual Scale): {rmse_actual}")

    # Calculate matching percentage
    logging.info("Calculating matching percentages...")
    # Avoid division by zero by replacing zeros in actual_prices with a very small number
    actual_prices_safe = np.where(actual_prices == 0, 1e-8, actual_prices)
    matching_percentage = 100 - (np.abs(actual_prices - predicted_prices) / actual_prices_safe) * 100
    matching_percentage = np.clip(matching_percentage, 0, 100)  # Ensure values are between 0 and 100

    # Create a DataFrame for results
    results_df = pd.DataFrame({
        'Date': dates_test,
        'Actual Close': actual_prices.flatten(),
        'Predicted Close': predicted_prices.flatten(),
        'Matching %': matching_percentage.flatten()
    })

    # Save predictions to CSV
    results_df.to_csv(predictions_csv_path, index=False)
    logging.info(f"Saved predictions with matching percentage to: {predictions_csv_path}")

    # Visualizing Overfitting Prevention

    # Plot actual vs predicted prices
    plot_predictions(actual_prices, predicted_prices, save_dir=charts_dir, sample_size=300)

    logging.info("Model training, evaluation, and visualization completed successfully.")


if __name__ == "__main__":
    main()
