import pandas as pd
from datetime import timedelta

# Function to detect the time interval in the input data
def detect_interval(data):
    # Calculate the difference between consecutive timestamps
    time_diffs = data['date'].diff().dropna()
    # Get the most common time difference (mode)
    mode_diff = time_diffs.mode()[0]
    return mode_diff

# Function to resample OHLC data to a specified candle size
def resample_data(input_file, output_file, candle_size):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Infer the datetime format automatically
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    # Sort data by date (if not already sorted)
    df = df.sort_values(by='date')
    
    # Detect the time interval of the input data
    detected_interval = detect_interval(df)
    print(f"Detected interval: {detected_interval}")

    # Convert detected interval to human-readable format
    if detected_interval < timedelta(minutes=1):
        print("Error: Detected interval is less than 1 minute. Please check the data.")
        return
    print(f"Input data appears to be {detected_interval.total_seconds() // 60:.0f}-minute candles.")
    
    # Set 'date' column as the index
    df.set_index('date', inplace=True)
    
    # Resample to the desired candle size
    resampled_df = df.resample(candle_size).agg({
        'open': 'first',          # First value in the interval
        'high': 'max',            # Maximum value in the interval
        'low': 'min',             # Minimum value in the interval
        'close': 'last',          # Last value in the interval
        'volume': 'sum'           # Sum of volumes in the interval
    }).dropna()  # Drop any rows with NaN values (e.g., incomplete intervals)

    # Save the resampled data to a new CSV file
    resampled_df.to_csv(output_file)
    print(f"Resampled data saved to {output_file}")

# Main function to drive the program
def main():
    input_file = "../data/NIFTY_100_minute.csv"  # Specify the path to the input CSV file here
    print("Supported candle sizes:")
    print("2T, 3T, 4T, 5T, 10T, 15T, 30T, 1H, 2H, 3H, 4H, 5H, 10H, 1D")
    candle_size = input("Enter the desired candle size (e.g., 5T for 5 minutes, 1H for 1 hour): ").strip()
    output_file = input("Enter the path to save the output CSV file: ")
    
    try:
        resample_data(input_file, output_file, candle_size)
    except Exception as e:
        print(f"Error: {e}")

# Run the program
if __name__ == "__main__":
    main()
