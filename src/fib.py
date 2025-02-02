import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# Function to identify swings
def identify_swings(df, n):
    df['Swing_High'] = df['high'] == df['high'].rolling(window=2 * n + 1, center=True).max()
    df['Swing_Low'] = df['low'] == df['low'].rolling(window=2 * n + 1, center=True).min()
    df['Swing_High'] = df['Swing_High'].fillna(False)
    df['Swing_Low'] = df['Swing_Low'].fillna(False)
    return df


# Function to evaluate performance of different 'n' values
def evaluate_n(df, n_values):
    results = []
    for n in n_values:
        temp_df = identify_swings(df.copy(), n)
        swing_highs = temp_df[temp_df['Swing_High']]
        swing_lows = temp_df[temp_df['Swing_Low']]

        num_swings = len(swing_highs) + len(swing_lows)
        avg_price_change = np.mean(
            np.abs(swing_highs['high'].diff().dropna().tolist() + swing_lows['low'].diff().dropna().tolist())
        )
        avg_duration = np.mean(
            pd.concat([swing_highs['date'], swing_lows['date']]).sort_values().diff().dropna().dt.total_seconds() / (
                    24 * 3600)
        )

        results.append({
            'n': n,
            'num_swings': num_swings,
            'avg_price_change': avg_price_change,
            'avg_duration_days': avg_duration
        })
    return pd.DataFrame(results)


def determine_trend(prev_swing_low, current_swing_high, next_price):
    """
    Determine trend based on price movement between swings and next price action.
    """
    if next_price < prev_swing_low['low']:
        return 'downtrend'
    elif next_price > current_swing_high['high']:
        return 'uptrend'
    else:
        # If the price is within the range, consider it sideways
        return 'sideways'


# Function to find the optimal n
def find_optimal_n(df, n_range=range(5, 51, 5)):
    evaluation_results = evaluate_n(df, n_range)

    # Normalize the metrics
    max_price_change = evaluation_results['avg_price_change'].max()
    max_num_swings = evaluation_results['num_swings'].max()
    max_duration = evaluation_results['avg_duration_days'].max()

    # Normalize for scoring (0-1 scale)
    evaluation_results['price_change_score'] = evaluation_results['avg_price_change'] / max_price_change
    evaluation_results['swing_count_score'] = 1 - (
                evaluation_results['num_swings'] / max_num_swings)  # Lower swings preferred
    evaluation_results['duration_score'] = 1 - (
                abs(evaluation_results['avg_duration_days'] - 0.75) / max_duration)  # Preferring ~0.75 days

    # Add penalty for too few swings (when n is too large)
    evaluation_results['penalty'] = np.where(evaluation_results['num_swings'] < 1000, -0.5, 0)

    # Adjusted composite score with new weights
    evaluation_results['composite_score'] = (
            0.3 * evaluation_results['price_change_score'] +  # Reduced to 30% weight on price change
            0.4 * evaluation_results['swing_count_score'] +  # Increased to 40% weight on swing count
            0.3 * evaluation_results['duration_score'] +  # Increased to 30% weight on duration
            evaluation_results['penalty']  # Apply penalty for very low swing counts
    )

    # Select the n with the highest composite score
    optimal_n = evaluation_results.loc[
        evaluation_results['composite_score'].idxmax()
    ]['n']

    # Ensure output directory exists
    output_dir = '../data/output/charts'
    os.makedirs(output_dir, exist_ok=True)

    # Save the evaluation plot
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.plot(evaluation_results['n'], evaluation_results['num_swings'], marker='o')
    plt.title('Number of Swings Detected')
    plt.xlabel('n (Window Size)')
    plt.ylabel('Number of Swings')

    plt.subplot(1, 3, 2)
    plt.plot(evaluation_results['n'], evaluation_results['avg_price_change'], marker='o', color='green')
    plt.title('Average Price Change Between Swings')
    plt.xlabel('n (Window Size)')
    plt.ylabel('Price Change')

    plt.subplot(1, 3, 3)
    plt.plot(evaluation_results['n'], evaluation_results['avg_duration_days'], marker='o', color='red')
    plt.title('Average Duration Between Swings (Days)')
    plt.xlabel('n (Window Size)')
    plt.ylabel('Duration (Days)')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'optimal_n_evaluation_updated.png')
    plt.savefig(chart_path)
    print(f"Updated optimal 'n' evaluation plot saved as '{chart_path}'")

    return int(optimal_n)


# Function to calculate Fibonacci retracement levels
def calculate_fibonacci_levels(high, low):
    diff = high - low
    levels = {
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff
    }
    return levels


# Main execution

def main():
    file_path = '../data/input/nifty_5_minute_600_days.csv'
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.dropna(subset=['high', 'low', 'close'], inplace=True)

    optimal_n = find_optimal_n(df)
    print(f"Optimal value of n: {optimal_n}")

    df = identify_swings(df, n=optimal_n)

    fibonacci_features = []
    swing_highs = df[df['Swing_High']]
    swing_lows = df[df['Swing_Low']]

    for i in range(1, len(swing_highs)):
        prev_swing_low_data = swing_lows[swing_lows['date'] < swing_highs.iloc[i]['date']]
        if prev_swing_low_data.empty:
            continue

        prev_swing_low = prev_swing_low_data.iloc[-1]
        current_swing_high = swing_highs.iloc[i]

        # Determine the next price after the current swing high for trend validation
        next_price_data = df[df['date'] > current_swing_high['date']]
        next_price = next_price_data['close'].iloc[0] if not next_price_data.empty else current_swing_high['close']

        # Enhanced trend detection
        trend_direction = determine_trend(prev_swing_low, current_swing_high, next_price)

        high_price = current_swing_high['high']
        low_price = prev_swing_low['low']
        fib_levels = calculate_fibonacci_levels(high_price, low_price)

        for level_name, level_price in fib_levels.items():
            fibonacci_features.append({
                'date': current_swing_high['date'],
                'swing_high': high_price,
                'swing_low': low_price,
                'fibonacci_level': level_name,
                'fibonacci_price': level_price,
                'price_diff': high_price - level_price,
                'trend_direction': trend_direction,
                'time_duration_days': (current_swing_high['date'] - prev_swing_low['date']).days
            })

    # Save the Fibonacci retracement levels
    output_csv_dir = '../data/output/csv'
    os.makedirs(output_csv_dir, exist_ok=True)

    fibonacci_df = pd.DataFrame(fibonacci_features)
    csv_path = os.path.join(output_csv_dir, 'fibonacci_retracement_levels_optimized_n.csv')
    fibonacci_df.to_csv(csv_path, index=False)

    print(f"Fibonacci retracement levels saved for model training at '{csv_path}'.")

if __name__ == "__main__":
    main()
