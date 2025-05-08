import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import yfinance as yf
import praw
import openpyxl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from functions import dataExploration, calc_indicators, get_reddit_data, get_guardian_data

#SEGMENT: Basic Information
index = "TSLA"
exchange = "^IXIC"
years = 5
end = datetime.today()
start = end - timedelta(days=365 * years)

df = yf.Ticker(index).history(start=start, end=end)

# SANITY CHECK: Verify the DataFrame has more than one row
assert len(df) > 1, "Error: The stock price DataFrame contains less than 2 rows. Check the API response."

#SEGMENT: Adding Exchange Data
#NOTE: Dow Jones : ^DJI || NYSE : NYA || NASDAQ : ^IXIC || AMEX : ^AMEX || S&P 500 : ^SPX
exchange_df = yf.Ticker(exchange).history(start=start, end=end)
df["Exchange_Open"] = exchange_df["Open"]
df["Exchange_High"] = exchange_df["High"]
df["Exchange_Low"] = exchange_df["Low"]
df["Exchange_Close"] = exchange_df["Close"]
df["Exchange_Volume"] = exchange_df["Volume"]

# SANITY CHECK: Ensure all exchange columns are present and have no missing values
for col in ["Exchange_Open", "Exchange_High", "Exchange_Low", "Exchange_Close", "Exchange_Volume"]:
    assert col in df.columns, f"Error: Column '{col}' is missing from the DataFrame."
    assert df[col].isnull().sum() == 0, f"Error: Missing values found in column '{col}'."


df= df.reset_index()
df["Date"] = df["Date"].dt.date

#SEGMENT: Adding Technical Indicators
df = calc_indicators(df, "Close")

# SANITY CHECK: Ensure technical indicators were added correctly
expected_indicator_columns = ["EMA_60", "RSI", "MACD", "MACD_Signal"]
for col in expected_indicator_columns:
    assert col in df.columns, f"Error: Expected technical indicator column '{col}' is missing."


#SEGMENT: Adding Reddit Data
df_reddit = get_reddit_data(years)
df_reddit['text'] = df_reddit['title'] + " " + df_reddit['selftext'].fillna("")

analyzer = SentimentIntensityAnalyzer()
df_reddit['reddit_sentiment'] = df_reddit['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
reddit_sentiment = df_reddit.groupby('Date').agg({
    'reddit_sentiment': 'mean',
    'reddit_score': 'sum'
}).reset_index()
# SANITY CHECK: Verify Reddit sentiment data has no missing values
assert reddit_sentiment.isnull().sum().sum() == 0, "Error: Missing values found in Reddit sentiment data."


# SEGMENT: Adding Guardian Data
df_guardian = get_guardian_data(years)
df_guardian["Date"] = pd.to_datetime(df_guardian["Date"]).dt.date
df_guardian['guardian_sentiment'] = df_guardian['GuardianTitle'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
guardian_sentiment = df_guardian.groupby('Date').agg({
    'guardian_sentiment': 'mean',
})
# SANITY CHECK: Verify Guardian sentiment data has no missing values
assert guardian_sentiment.isnull().sum().sum() == 0, "Error: Missing values found in Guardian sentiment data."

# SEGMENT: Adding NYT Data
df_nyt = pd.read_csv('nyt.csv')
df_nyt["Date"] = pd.to_datetime(df_nyt["pub_date"]).dt.date
df_nyt['text'] = df_nyt['text'].fillna("")
df_nyt['nyt_sentiment'] = df_nyt['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
nyt_sentiment = df_nyt.groupby('Date').agg({
    'nyt_sentiment': 'mean'
}).reset_index()
# SANITY CHECK: Verify NYT sentiment data has no missing values
assert nyt_sentiment.isnull().sum().sum() == 0, "Error: Missing values found in NYT sentiment data."


# SEGMENT: Merging Data & Preprocessing
data = df.merge(reddit_sentiment, on="Date", how="left")
data = data.merge(guardian_sentiment, on="Date", how="left")
data = data.merge(nyt_sentiment, on="Date", how="left")

data["reddit_sentiment"] = data["reddit_sentiment"].fillna(0)
data["reddit_score"] = data["reddit_score"].fillna(0)
data["guardian_sentiment"] = data["guardian_sentiment"].fillna(0)
data["nyt_sentiment"] = data["nyt_sentiment"].fillna(0)

data = data.drop(["Dividends", "Stock Splits"], axis=1)

data = data.fillna(0)


# SANITY CHECK: Verify no missing values remain in the final DataFrame
assert data.isnull().sum().sum() == 0, "Error: Missing values found in the final DataFrame."

# SANITY CHECK: Verify the number of rows is reasonable (years-1 * 252 : As we will be removing a few starting trading days to account for EMA_60 Calculation)
expected_min_rows = years-1 * 252  # Approximate trading days per year
assert len(data) >= expected_min_rows, f"Error: Final DataFrame has fewer rows ({len(data)}) than expected ({expected_min_rows})."

# SANITY CHECK: Manually verify the final DataFrame has the expected columns
print(data)
print(data.info())

# Save the DataFrame to a CSV file
data.to_csv("dataBackUp/final_data.csv")

# Save the DataFrame to a Parquet file
df.to_parquet('dataBackUp/final_data.parquet', engine='pyarrow', index=False)

# Save the DataFrame to a Excel file
data.to_excel('dataBackUp/final_data.xlsx', index=False)
