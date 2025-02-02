import numpy as np
import pandas as pd
import talib as talib


def calculate_atr(df, period=10):
    # Create an explicit copy to avoid modifying a view
    df = df.copy()

    # Calculate the three components of True Range using .loc for clarity
    df.loc[:, 'H-L'] = df.loc[:, 'high'] - df.loc[:, 'low']
    df.loc[:, 'H-PC'] = (df.loc[:, 'high'] - df.loc[:, 'close'].shift(1)).abs()
    df.loc[:, 'L-PC'] = (df.loc[:, 'low'] - df.loc[:, 'close'].shift(1)).abs()

    # True Range is the maximum of these three components
    df.loc[:, 'TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    # ATR is the simple moving average of TR over the specified period
    df.loc[:, 'ATR'] = df.loc[:, 'TR'].rolling(window=period, min_periods=1).mean()

    # Drop intermediate columns
    df.drop(['H-L', 'H-PC', 'L-PC'], axis=1, inplace=True)

    return df


def supertrend(df, period=10, multiplier=3):
    """
    Calculate the SuperTrend indicator.

    Parameters:
      df (DataFrame): Must contain columns: 'High', 'Low', 'Close'
      period (int): Lookback period for ATR calculation.
      multiplier (float): Multiplier for the ATR in band calculation.

    Returns:
      df (DataFrame): The input DataFrame with added columns:
                      'ATR', 'Basic Upper Band', 'Basic Lower Band',
                      'Final Upper Band', 'Final Lower Band', 'SuperTrend',
                      and 'Trend' (with values 'up' or 'down')
    """
    df = calculate_atr(df, period)

    # Calculate basic bands
    hl2 = (df['high'] + df['low']) / 2
    df['Basic Upper Band'] = hl2 + (multiplier * df['ATR'])
    df['Basic Lower Band'] = hl2 - (multiplier * df['ATR'])

    # Initialize final bands with the basic bands
    df['Final Upper Band'] = df['Basic Upper Band']
    df['Final Lower Band'] = df['Basic Lower Band']

    # Iterate through DataFrame rows to calculate final bands
    for i in range(1, len(df)):
        # Final Upper Band
        if (df['Basic Upper Band'].iloc[i] < df['Final Upper Band'].iloc[i - 1]) or \
                (df['close'].iloc[i - 1] > df['Final Upper Band'].iloc[i - 1]):
            df.at[df.index[i], 'Final Upper Band'] = df['Basic Upper Band'].iloc[i]
        else:
            df.at[df.index[i], 'Final Upper Band'] = df['Final Upper Band'].iloc[i - 1]

        # Final Lower Band
        if (df['Basic Lower Band'].iloc[i] > df['Final Lower Band'].iloc[i - 1]) or \
                (df['close'].iloc[i - 1] < df['Final Lower Band'].iloc[i - 1]):
            df.at[df.index[i], 'Final Lower Band'] = df['Basic Lower Band'].iloc[i]
        else:
            df.at[df.index[i], 'Final Lower Band'] = df['Final Lower Band'].iloc[i - 1]

    # Initialize SuperTrend column
    df['SuperTrend'] = np.nan

    # Set the first value of SuperTrend as the first Final Upper Band or Lower Band based on close
    # (You could also leave it as NaN for the first bar)
    if df['close'].iloc[0] <= df['Final Upper Band'].iloc[0]:
        df.at[df.index[0], 'SuperTrend'] = df['Final Upper Band'].iloc[0]
    else:
        df.at[df.index[0], 'SuperTrend'] = df['Final Lower Band'].iloc[0]

    # Now, determine SuperTrend for subsequent bars
    for i in range(1, len(df)):
        prev_st = df['SuperTrend'].iloc[i - 1]
        curr_close = df['close'].iloc[i]
        curr_final_upper = df['Final Upper Band'].iloc[i]
        curr_final_lower = df['Final Lower Band'].iloc[i]

        if prev_st == df['Final Upper Band'].iloc[i - 1]:
            if curr_close <= curr_final_upper:
                df.at[df.index[i], 'SuperTrend'] = curr_final_upper
            else:
                df.at[df.index[i], 'SuperTrend'] = curr_final_lower
        elif prev_st == df['Final Lower Band'].iloc[i - 1]:
            if curr_close >= curr_final_lower:
                df.at[df.index[i], 'SuperTrend'] = curr_final_lower
            else:
                df.at[df.index[i], 'SuperTrend'] = curr_final_upper
        else:
            # Fallback if previous SuperTrend is not defined as one of the bands
            if curr_close <= curr_final_upper:
                df.at[df.index[i], 'SuperTrend'] = curr_final_upper
            else:
                df.at[df.index[i], 'SuperTrend'] = curr_final_lower

    # Optional: define the trend based on where the close is relative to the SuperTrend
    df['STX'] = np.where(df['close'] > df['SuperTrend'], 'up', 'down')

    return df


