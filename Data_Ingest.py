import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import yfinance as yf
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from functions import dataExploration, calc_indicators, get_reddit_data, get_guardian_data

#SEGMENT: Basic Information
index = "TSLA"
exchange = "^IXIC"
years = 1
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

#SEGMENT: Adding Technical Indicators
df = calc_indicators(df, "Close")

#SEGMENT: Adding Reddit Data
df_reddit = get_reddit_data(years)
df_reddit['text'] = df_reddit['title'] + " " + df_reddit['selftext'].fillna("")

analyzer = SentimentIntensityAnalyzer()
df_reddit['reddit_sentiment'] = df_reddit['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
reddit_sentiment = df_reddit.groupby('Date').agg({
    'reddit_sentiment': 'mean',
    'reddit_score': 'sum'
}).reset_index()

# SEGMENT: Adding Guardian Data
df_guardian = get_guardian_data(years)
df_guardian["Date"] = pd.to_datetime(df_guardian["Date"]).dt.date
df_guardian['guardian_sentiment'] = df_guardian['GuardianTitle'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
guardian_sentiment = df_guardian.groupby('Date').agg({
    'guardian_sentiment': 'mean',
})

# SEGMENT: Adding NYT Data
df_nyt = pd.read_csv('nyt.csv')
df_nyt["Date"] = pd.to_datetime(df_nyt["pub_date"]).dt.date
df_nyt['text'] = df_nyt['text'].fillna("")
df_nyt['nyt_sentiment'] = df_nyt['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
nyt_sentiment = df_nyt.groupby('Date').agg({
    'nyt_sentiment': 'mean'
}).reset_index()

print(nyt_sentiment.info())
print(df.info())