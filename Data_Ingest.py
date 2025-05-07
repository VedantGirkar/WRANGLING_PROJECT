import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import yfinance as yf
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#SEGMENT: Basic Information
index = "TSLA"
exchange = "^IXIC"
years = 5
end = datetime.today()
start = end - timedelta(days=365 * years)

df = yf.Ticker(index).history(start=start, end=end)

#SEGMENT: Adding Exchange Data
#NOTE: Dow Jones : ^DJI || NYSE : NYA || NASDAQ : ^IXIC || AMEX : ^AMEX || S&P 500 : ^SPX
exchange_df = yf.Ticker(exchange).history(start=start, end=end)
df["Exchange_Open"] = exchange_df["Open"]
df["Exchange_High"] = exchange_df["High"]
df["Exchange_Low"] = exchange_df["Low"]
df["Exchange_Close"] = exchange_df["Close"]
df["Exchange_Volume"] = exchange_df["Volume"]

df= df.reset_index()
df["Date"] = df["Date"].dt.date

print(df)
print(df.info())