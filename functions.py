import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

def dataExploration(df, y):
    print("--------------------------------------------------------------------------------")
    x = df[y].unique()
    print("DATA EXPLORATION")
    print("Data Exploration for : ", y)
    print("Total Values : ", len(df[y]))
    print("Count : ", (len(df[y]) - df[y].isna().sum()))
    print("Null Values : ", df[y].isna().sum())
    print("Number of Uniques : ", len(x))
    print("Unique Values : ", df[y].unique())
    print(df[y].value_counts())
    print(type(y))
    print("--------------------------------------------------------------------------------")

def calc_indicators(df, column):
    """
    Calculate various technical indicators for stock market analysis.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data with columns ['Close', 'High', 'Low', 'Volume'].

    Returns:
        pd.DataFrame: Updated DataFrame with calculated indicators as new columns.
    """

    # Ensure the DataFrame has necessary columns
    required_columns = {'Close', 'High', 'Low', 'Volume'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Exponential Moving Averages (EMA)
    df['EMA_7'] = df[column].ewm(span=7, adjust=False).mean()  # Short-term trend
    df['EMA_30'] = df[column].ewm(span=30, adjust=False).mean()  # Medium-term trend
    df['EMA_60'] = df[column].ewm(span=60, adjust=False).mean()  # Long-term trend

    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df[column].ewm(span=12, adjust=False).mean() - df[column].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line for MACD

    # RSI (Relative Strength Index)
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))  # Measures overbought/oversold conditions

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)  # Momentum indicator

    # Rate of Change (ROC)
    for period in [1, 2, 7, 30, 60]:
        df[f'ROC_{period}'] = df[column].pct_change(periods=period) * 100  # % change in price over period

    # Bollinger Bands
    df['Bollinger_Middle_Band'] = df[column].rolling(window=20).mean().shift(1)  # Central moving average
    df['Bollinger_Upper_Band'] = df['Bollinger_Middle_Band'] + 2 * df[column].rolling(window=20).std()  # Upper volatility band
    df['Bollinger_Lower_Band'] = df['Bollinger_Middle_Band'] - 2 * df[column].rolling(window=20).std()  # Lower volatility band

    # Average True Range (ATR)
    tr = pd.concat(
        [df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())],
        axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()  # Measures market volatility

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df[column].diff()) * df['Volume']).fillna(0).cumsum()  # Tracks buying/selling pressure

    # VWAP (Volume Weighted Average Price)
    df['TP'] = (df["Open"] + df['High'] + df['Low'] + df['Close']) / 4
    df["TP"] = df["TP"].shift(1)  # Typical price
    df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()  # Weighted average price

    # TWAP (Time Weighted Average Price)
    df['TWAP'] = df[column].rolling(window=20).mean().shift(1)  # Simple time-weighted average

    # Chaikin Money Flow (CMF)
    mf = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['CMF'] = mf.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()  # Money flow into/out of asset

    # Advance-Decline Line (Custom for Tesla vs Nasdaq)
    df['Advance_Decline'] = np.where(df[column] > df[column].shift(), 1,
                                     -1).cumsum()  # Tracks cumulative advances/declines

    # Cumulative Volume Index (CVI)
    df['CVI'] = df['Volume'].cumsum()  # Tracks cumulative trading volume

    return df