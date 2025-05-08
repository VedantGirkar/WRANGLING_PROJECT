import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load the dataset
data = pd.read_csv('/Users/tanvishakose/Downloads/final_data (2).csv', parse_dates=['Date'])
data.sort_values('Date', inplace=True)

# Filter the data for only the last 1 year
one_year_data = data[data['Date'] >= (data['Date'].max() - pd.DateOffset(years=1))]

# Apply rolling average for smoothing (7-day window)
one_year_data['nyt_sentiment_smooth'] = one_year_data['nyt_sentiment'].rolling(window=7).mean()
one_year_data['macd_smooth'] = one_year_data['MACD'].rolling(window=7).mean()

# Compute correlation
correlation = one_year_data[['nyt_sentiment_smooth', 'macd_smooth']].corr().iloc[0, 1]

# --- Visualization ---
fig = go.Figure()

# Plot smoothed NYT sentiment
fig.add_trace(go.Scatter(
    x=one_year_data['Date'], 
    y=one_year_data['nyt_sentiment_smooth'], 
    mode='lines', 
    name='NYT Sentiment (Smoothed)', 
    line=dict(color='blue')
))

# Plot smoothed MACD
fig.add_trace(go.Scatter(
    x=one_year_data['Date'], 
    y=one_year_data['macd_smooth'], 
    mode='lines', 
    name='MACD (Smoothed)', 
    line=dict(color='green'),
    yaxis='y2'
))

# Update layout for dual y-axis
fig.update_layout(
    title=f"NYT Sentiment vs. MACD Over the Last Year (Correlation: {correlation:.2f})",
    xaxis_title="Date",
    yaxis_title="NYT Sentiment",
    yaxis2=dict(title="MACD", overlaying='y', side='right'),
    legend_title="Metrics",
    template="plotly_white"
)

# Display the plot
fig.show()