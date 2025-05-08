import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
df = pd.read_csv("/Users/tanvishakose/Downloads/final_data.csv") 
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])

# Filter for March 24–27, 2025
start_date = pd.to_datetime("2025-03-24")
end_date = start_date + pd.Timedelta(days=3)
march_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
march_df['Daily_Return_%'] = march_df['Close'].pct_change() * 100

# Create 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

# --- Candlestick Plot ---
for _, row in march_df.iterrows():
    date_num = mdates.date2num(row['Date'])
    color = 'green' if row['Close'] >= row['Open'] else 'red'

    # Wick
    ax1.plot([row['Date'], row['Date']], [row['Low'], row['High']], color='black')

    # Body
    rect = Rectangle((date_num - 0.2, min(row['Open'], row['Close'])),
                     0.4,
                     abs(row['Close'] - row['Open']),
                     color=color)
    ax1.add_patch(rect)

ax1.set_ylabel("Price ($)")
ax1.set_title("Tesla: Return %, and NYT FinBERT Sentiment (March 24–27, 2025)")
ax1.grid(True)

# --- Daily Return Plot ---
ax2.bar(march_df['Date'], march_df['Daily_Return_%'], color='skyblue')
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel("Return (%)")
ax2.grid(True)

# --- FinBERT Sentiment Plot ---
ax3.plot(march_df['Date'], march_df['NYT_finbert_score'], marker='o', linestyle='--', color='red')
ax3.set_ylabel("FinBERT Sentiment")
ax3.set_xlabel("Date")
ax3.grid(True)

# Format x-axis
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()