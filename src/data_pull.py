from kiteconnect import KiteConnect
import datetime
import pandas as pd


# Replace with your API key and API secret
api_key = "qb4d9rcg47ghiqjx"
api_secret = "e697seho1m1rrt8zmp1xijf1u8e6jz9k"
request_token = "jGFAlwTGFE6K3u3jf67jA94ZNdQRtQW2"

kite = KiteConnect(api_key=api_key)

# Exchange request token for access token
data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]

print("Access Token:", access_token)

# Instrument token for NIFTY 50 (Replace with correct instrument if needed)
instrument_token = 256265
interval = "5minute"

# Instrument token for NIFTY 50 (Replace if needed)
instrument_token = 256265
interval = "5minute"

# Calculate date range
end_date = datetime.date.today() - datetime.timedelta(days=1)  # Yesterday
start_date = end_date - datetime.timedelta(days=60)  # Last 60 days from yesterday

# Fetch historical data
historical_data = kite.historical_data(
    instrument_token, start_date, end_date, interval
)

# Convert to DataFrame
df = pd.DataFrame(historical_data)

# Save DataFrame to CSV file
csv_filename = "nifty_5_minute.csv"
df.to_csv(csv_filename, index=False)

print(f"Historical data saved successfully to {csv_filename}")
