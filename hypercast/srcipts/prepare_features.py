# scripts/prepare_features.py
import pandas as pd
import numpy as np
from datetime import datetime

def create_features(df):
    """
    Transform raw ERCOT data into ML-ready features
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Lag features (previous prices)
    df['price_lag_1h'] = df['price'].shift(1)
    df['price_lag_24h'] = df['price'].shift(24)
    df['price_lag_168h'] = df['price'].shift(168)  # 1 week
    
    # Rolling statistics
    df['price_rolling_mean_24h'] = df['price'].rolling(24).mean()
    df['price_rolling_std_24h'] = df['price'].rolling(24).std()
    
    # Load features (if you have load data)
    if 'load' in df.columns:
        df['load_lag_1h'] = df['load'].shift(1)
        df['load_rolling_mean_24h'] = df['load'].rolling(24).mean()
    
    # Drop rows with NaN from lag features
    df = df.dropna()
    
    return df

def create_sequences(df, lookback=168, horizon=24):
    """
    Create sequences for time-series forecasting
    lookback: hours of history to use (168 = 1 week)
    horizon: hours ahead to forecast (24 = 1 day)
    """
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'price']]
    
    X, y = [], []
    
    for i in range(lookback, len(df) - horizon):
        # Input: past 'lookback' hours of features
        X.append(df[feature_cols].iloc[i-lookback:i].values)
        # Target: next 'horizon' hours of prices
        y.append(df['price'].iloc[i:i+horizon].values)
    
    return np.array(X), np.array(y)

# Usage
if __name__ == "__main__":
    df = pd.read_csv('data/processed/ercot_merged.csv')
    df_features = create_features(df)
    df_features.to_csv('data/processed/ercot_features.csv', index=False)
    
    X, y = create_sequences(df_features, lookback=168, horizon=24)
    np.save('data/processed/X_train.npy', X)
    np.save('data/processed/y_train.npy', y)
    print(f"Created {len(X)} sequences")