import pandas as pd
import plotly.graph_objects as go
import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DatetimeProperties.to_pydatetime")

# Load and filter data
df = pd.read_csv("final_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df[(df['Date'] >= '2023-10-07') & (df['Date'] <= '2023-11-01')]

# Convert Date column properly (Recommended way)
df['Date'] = df['Date'].to_numpy()

# Create the figure
fig = go.Figure()

# Add the line plot for Closing Price
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines+markers+text',
    name='Closing Price',
    line=dict(color='black', width=2),
    marker=dict(color='black'),
    text=[f'{price:.0f}' for price in df['Close']],
    textposition='top center',
    textfont=dict(
        family='Times New Roman',
        size=10,
        color='black'
    )
))

# Update layout to match your specifications
fig.update_layout(
    title="Tesla Stock Price with Reddit Sentiment<br>(7th Oct 2023 - 1st Nov 2023)",
    title_font=dict(size=16, family='Times New Roman', color='black'),
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    font=dict(family="Times New Roman", size=12, color='black'),
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=40),
    hovermode="x unified",
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=1,
        ticks='outside',
        tickfont=dict(
            family='Times New Roman',
            size=12,
            color='black'
        )
    ),
    yaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=1,
        ticks='outside',
        tickfont=dict(
            family='Times New Roman',
            size=12,
            color='black'
        )
    ),
    legend=dict(
        y=1,
        x=1,
        font=dict(
            family='Times New Roman',
            size=12,
            color='black'
        ),
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)'
    )
)

# Show the interactive plot
fig.show()