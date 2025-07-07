import yfinance as yf
import pandas as pd
import datetime as dt
import os

# --- Configuration ---
# Your list of stock tickers
TICKERS = ['NVDA', 'META', 'MSFT', 'AMZN', 'AAPL', 'PLTR', 'UBER', 'DASH', 'GOOGL', 'CRWD', 'TSLA', '^GSPC', '^IXIC', 'BTC-USD']

# Sector ETF (e.g., XLK for tech sector)
SECTOR_ETF = 'XLK'

# Number of past calendar days to pull data for (approximate, adjust for weekends/holidays)
# UPDATED: Changed from 90 to 180 days
DAYS_TO_PULL = 180

# Path for your CSV file to store daily metrics
CSV_FILE_PATH = 'historical_daily_metrics.csv'

# --- End Configuration ---

def fetch_data(tickers, days_to_pull):
    """
    Fetches historical price and volume data for a list of tickers.
    Returns a dictionary of pandas DataFrames.
    """
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days_to_pull)

    data = {}
    for ticker in tickers:
        try:
            # Download data for the specified period
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                data[ticker] = df
                print(f"Successfully fetched data for {ticker}")
            else:
                print(f"No data fetched for {ticker}. Skipping.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return data

def calculate_metrics(df, sector_df=None):
    """
    Computes various financial metrics for a given stock DataFrame.
    Optionally computes relative return if sector_df is provided.
    """
    if df.empty:
        print("Input DataFrame is empty. Skipping metric calculation.")
        return pd.DataFrame()

    # If yfinance returns a MultiIndex for columns (e.g., ('Close', 'NVDA')), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if 'Close' not in df.columns or 'Volume' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
        print(f"Error: Missing essential columns in DataFrame after potential column flattening. Columns: {df.columns.tolist()}")
        return pd.DataFrame()

    df = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # 1. Close (already present)
    # 2. 20-day rolling high (highest close over the last 20 trading days)
    df['RollingHigh_20'] = df['Close'].rolling(window=20).max()

    # 3. Drawdown_20 = (Close – RollingHigh_20) / RollingHigh_20
    numerator = df['Close'] - df['RollingHigh_20']
    denominator = df['RollingHigh_20']
    df['Drawdown_20'] = numerator / denominator

    # 4. ATR_14 (14-day Average True Range)
    df['High_Low'] = df['High'] - df['Low']
    df['High_PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    df['ATR_14'] = df['TR'].ewm(span=14, adjust=False).mean()

    # 5. DailyPctDrop = (Close(t‐1) – Close(t)) / Close(t‐1)
    df['DailyPctDrop'] = (df['Close'].shift(1) - df['Close']) / df['Close'].shift(1)

    # 6. RSI_14 (14-day Relative Strength Index)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 7. AvgVolume_20 = 20-day simple rolling average of Volume
    df['AvgVolume_20'] = df['Volume'].rolling(window=20).mean()

    # 8. RVOL_20 = Volume / AvgVolume_20
    df['RVOL_20'] = df['Volume'] / df['AvgVolume_20']

    # 9. SectorClose (e.g., XLK’s Close)
    if sector_df is not None and not sector_df.empty and 'Close' in sector_df.columns:
        if isinstance(sector_df.columns, pd.MultiIndex):
            sector_df.columns = [col[0] if isinstance(col, tuple) else col for col in sector_df.columns]
        df['SectorClose'] = sector_df['Close'].reindex(df.index, method='ffill')
    else:
        df['SectorClose'] = pd.NA

    # 10. StockReturn and SectorReturn (DailyPctChange is more common)
    df['StockReturn'] = df['Close'].pct_change()

    if sector_df is not None and not sector_df.empty and 'Close' in sector_df.columns:
        df['SectorReturn'] = df['SectorClose'].pct_change()
        # 11. RelReturn = StockReturn – SectorReturn
        df['RelReturn'] = df['StockReturn'] - df['SectorReturn']
    else:
        df['SectorReturn'] = pd.NA
        df['RelReturn'] = pd.NA

    output_columns = [
        'Close',
        'RollingHigh_20',
        'Drawdown_20',
        'ATR_14',
        'DailyPctDrop',
        'RSI_14',
        'AvgVolume_20',
        'RVOL_20',
        'SectorClose',
        'StockReturn',
        'SectorReturn',
        'RelReturn'
    ]
    return df[output_columns].dropna()

# --- Main Script Execution ---

print("Starting daily stock metrics script...")

# 1. Fetch data
all_tickers_to_fetch = TICKERS + [SECTOR_ETF]
raw_data = fetch_data(all_tickers_to_fetch, DAYS_TO_PULL)

stock_data = {ticker: raw_data[ticker] for ticker in TICKERS if ticker in raw_data}
sector_etf_df = raw_data.get(SECTOR_ETF)

if not stock_data:
    print("No stock data available. Exiting script.")
    exit()
if sector_etf_df is None or sector_etf_df.empty:
    print(f"Warning: Could not fetch data for {SECTOR_ETF}. Relative return calculations will be skipped.")
    sector_etf_df = None

# 2. Calculate metrics for each ticker
all_metrics_df = {}
for ticker, df in stock_data.items():
    print(f"\nCalculating metrics for {ticker}...")
    metrics_df = calculate_metrics(df, sector_df=sector_etf_df)
    if not metrics_df.empty:
        all_metrics_df[ticker] = metrics_df
    else:
        print(f"No metrics calculated for {ticker}. Skipping.")

if not all_metrics_df:
    print("No metrics calculated for any stock. Exiting script.")
    exit()

# 3. Prepare ALL historical data for CSV
full_historical_df_list = []
for ticker, metrics_df in all_metrics_df.items():
    if metrics_df.empty:
        continue
    
    metrics_df_with_ticker = metrics_df.copy()
    metrics_df_with_ticker['Ticker'] = ticker
    
    metrics_df_with_ticker = metrics_df_with_ticker.reset_index()
    metrics_df_with_ticker = metrics_df_with_ticker.rename(columns={'index': 'Date'})

    full_historical_df_list.append(metrics_df_with_ticker)

if not full_historical_df_list:
    print("No historical data prepared for CSV. Exiting script.")
    exit()

final_historical_df = pd.concat(full_historical_df_list, ignore_index=True)

standard_metrics_cols = [
    'Close', 'RollingHigh_20', 'Drawdown_20', 'ATR_14', 'DailyPctDrop',
    'RSI_14', 'AvgVolume_20', 'RVOL_20', 'SectorClose', 'StockReturn',
    'SectorReturn', 'RelReturn'
]
desired_final_cols = ['Date', 'Ticker'] + standard_metrics_cols
final_historical_df = final_historical_df[desired_final_cols]

print("\n--- Full Historical Daily Metrics (First 5 rows) ---")
print(final_historical_df.head())
print(f"Total rows in historical data: {len(final_historical_df)}")
print("----------------------------------------------------\n")

# 4. Save to CSV
try:
    final_historical_df.to_csv(CSV_FILE_PATH, mode='w', header=True, index=False)
    print(f"Saved {len(final_historical_df)} historical daily metrics to '{CSV_FILE_PATH}'.")

except Exception as e:
    print(f"Error saving data to CSV: {e}")

print("Script finished successfully.")