import plotly.express as px
import pandas as pd

# Load your CSV file
df = pd.read_csv("/Users/tanvishakose/Downloads/final_data.csv")

# Select only numeric columns
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = df_numeric.corr()

# Plot the heatmap
fig = px.imshow(
    correlation_matrix,
    labels=dict(color="Correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    color_continuous_scale='RdBu'
)

# Remove title
fig.update_layout(title=None)

# Move colorbar even closer
fig.update_layout(
    coloraxis_colorbar=dict(
        x=0.69,
        thickness=15,
        title="Correlation",
        tickfont=dict(size=12),
        titlefont=dict(size=14)
    )
)

# Axis styling
fig.update_xaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside', showline=True)
fig.update_yaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside', showline=True)

# Show the figure
fig.show()
