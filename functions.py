import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import praw
import requests

def dataExploration(df, y):
    # SANITY CHECK: Ensure the column exists in the DataFrame
    assert y in df.columns, f"Error: Column '{y}' not found in the DataFrame."
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
    # SANITY CHECK: Ensure the DataFrame has necessary columns
    required_columns = {'Close', 'High', 'Low', 'Volume'}
    missing_columns = required_columns - set(df.columns)
    assert not missing_columns, f"Error: Missing required columns: {missing_columns}"

    # SANITY CHECK: Ensure the specified column exists
    assert column in df.columns, f"Error: Column '{column}' not found in the DataFrame."


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

def get_reddit_data(years):
    reddit = praw.Reddit(
        client_id="T7oKFrZVFHPIJtr5M67JIg",  # Your client ID
        client_secret="KmJuy0s55nO87JRajBxHC2SeBLVeyQ",  # Your client secret
        user_agent="tesla-sentiment-script by u/Optimal-Bar-2756"  # Your user agent
    )

    subreddit_keywords = {
        "tesla": ["crash", "fire", "stocks", "lithium shortage"],
        "stocks": ["tesla", "elon musk", "lithium shortage"],
        "stockmarket": ["tesla", "elon musk", "lithium shortage"],
        "news": ["tesla", "elon musk", "lithium shortage"],
        "worldnews": ["tesla", "elon musk", "lithium shortage"]
    }

    posts = []
    end = datetime.today()
    start = end - timedelta(days=years * 365)

    for subreddit_name, keywords in subreddit_keywords.items():
        subreddit = reddit.subreddit(subreddit_name)
        for keyword in keywords:
            for submission in subreddit.search(keyword, sort="new", time_filter="all", limit=1000):
                # Only collect posts within the last 5 years
                # post_date = fromtimestamp(submission.created_utc, tz=dt.timezone.utc).date()
                post_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).date()
                if post_date >= start.date():
                    posts.append({
                        "Date": post_date,
                        "subreddit": subreddit_name,
                        "keyword": keyword,
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "reddit_score": submission.score
                    })
    # SANITY CHECK: Ensure the posts list is not empty
    assert len(posts) > 0, "Error: No Reddit posts were fetched."

    df = pd.DataFrame(posts)

    # SANITY CHECK: Verify the DataFrame has the expected columns
    expected_columns = {"Date", "subreddit", "keyword", "title", "selftext", "reddit_score"}
    missing_columns = expected_columns - set(df.columns)
    assert not missing_columns, f"Error: Missing expected columns in Reddit DataFrame: {missing_columns}"

    return df

def get_guardian_data(years):
    # Define API parameters
    api_key = "6463ff44-686e-4a55-a25d-2d7df802a577"
    search_query = 'tesla'
    sort = "newest"
    page_size = 50
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    end = end.strftime("%Y%m%d")
    start = start.strftime("%Y%m%d")
    fields = "body"
    tag = "technology"

    # Initialize an empty list to store all articles
    all_articles = []

    # Loop through multiple pages
    for page in range(1, 50):  # Adjust the range to include more pages (e.g., 1 to 10)
        url = (f"https://content.guardianapis.com/search?page={page}&"
               f"q={search_query}&"
               f"page-size={page_size}&"
               f"order-by={sort}&"
               f"api-key={api_key}&"
               f"query-fields={fields}&"
               )

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data['response']['results']

            # Append each article to the list
            for article in articles:
                all_articles.append({
                    'Date': article['webPublicationDate'],
                    'GuardianTitle': article['webTitle']
                })
        else:
            print(f"Error on page {page}: {data['message']}")
            break  # Stop the loop if there's an error

    # SANITY CHECK: Ensure the articles list is not empty
    assert len(all_articles) > 0, "Error: No Guardian articles were fetched."

    df = pd.DataFrame(all_articles)

    # SANITY CHECK: Verify the DataFrame has the expected columns
    expected_columns = {"Date", "GuardianTitle"}
    missing_columns = expected_columns - set(df.columns)
    assert not missing_columns, f"Error: Missing expected columns in Guardian DataFrame: {missing_columns}"

    return df