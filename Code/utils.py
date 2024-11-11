
import yfinance as yf
import pandas as pd
def download_and_prepare_data():
    try:
        # Download data
        ticker = 'AAPL'
        start_date = '2020-01-01'
        end_date = '2023-01-01'

        print(f"Downloading {ticker} data...")
        df = yf.download(ticker, start=start_date, end=end_date)

        # Fix the column structure
        df = df.droplevel(1, axis=1)
        df.columns = df.columns.str.lower()
        df = df.drop('adj close', axis=1)

        # Create features
        print("Preparing features...")
        df["feature_close"] = df["close"].pct_change()
        df["feature_open"] = df["open"] / df["close"]
        df["feature_high"] = df["high"] / df["close"]
        df["feature_low"] = df["low"] / df["close"]
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7).max()

        # Clean data
        df.dropna(inplace=True)
        df.index.name = None

        print(f"Data shape: {df.shape}")
        print("\nColumns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        return df

    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise
