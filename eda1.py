import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("/Users/tanvishakose/Downloads/final_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.tail(30)
df_candle = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
df_candle.set_index('Date', inplace=True)

# Custom style
custom_style = mpf.make_mpf_style(
    base_mpf_style='classic',
    rc={
        'font.family': 'Times New Roman',
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    },
    marketcolors=mpf.make_marketcolors(
        up='#2ca02c',
        down='#d62728',
        edge='inherit',
        wick='gray',
        volume='lightblue'
    ),
    gridcolor='lightgray',
    gridstyle='--',
    facecolor='white',
    figcolor='white'
)

# Overlays
apds = [
    mpf.make_addplot(df['Bollinger_Upper_Band'], color='darkgreen', linestyle='--', width=1.2),
    mpf.make_addplot(df['Bollinger_Lower_Band'], color='darkred', linestyle='--', width=1.2),
    mpf.make_addplot(df['ATR'], panel=1, color='purple', width=1.3, ylabel='ATR'),
    mpf.make_addplot(df['reddit_sentiment'], panel=2, color='navy', width=1.3, ylabel='Reddit Sentiment')
]

# Plot with figure/axes
fig, axes = mpf.plot(
    df_candle,
    type='candle',
    style=custom_style,
    volume=True,
    addplot=apds,
    ylabel='Price ($)',
    ylabel_lower='Volume',
    panel_ratios=(3, 1, 1),
    figratio=(16, 9),
    figscale=1.5,
    returnfig=True,
    datetime_format='%b %d'
)

# Centered title on top panel
axes[0].text(
    0.5, 0.95,
    'Tesla Volatility with Bollinger Bands, ATR & Reddit Sentiment',
    transform=axes[0].transAxes,
    fontsize=14,
    fontweight='bold',
    fontfamily='Times New Roman',
    verticalalignment='top',
    horizontalalignment='center'
)

plt.show()