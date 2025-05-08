import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

df = pd.read_csv('final_data.csv')

# SANITY CHECK: Ensure the DataFrame is not empty
assert not df.empty, "Error: The loaded DataFrame is empty."

df = df.fillna(0)
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].shift(1)

df_base = df.filter(items=['Date', 'High', 'Low', 'Close', 'Volume', "Open"])
df_technical = df.drop(columns=['reddit_sentiment', 'guardian_sentiment', 'reddit_score', 'nyt_sentiment'])

# SANITY CHECK: Ensure subsets contain the expected columns
expected_base_columns = {'Date', 'High', 'Low', 'Close', 'Volume', 'Open'}
missing_base_columns = expected_base_columns - set(df_base.columns)
assert not missing_base_columns, f"Error: Missing expected columns in df_base: {missing_base_columns}"


def random_forest_regression(df, dataset_name):
    # Step 1: Data Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # SANITY CHECK: Ensure 'Date' column is successfully converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(df.index), "Error: 'Date' column is not properly converted to datetime."

    # Create target variable (next day's High)
    df['Target'] = df['High'].shift(-1)  # Predict next day's High

    # Feature engineering
    features = df.drop(columns=['High', 'Target'])
    target = df['Target']

    # Handle missing values after shifting
    df.dropna(subset=['Target'], inplace=True)

    # Split into train and test (last 30 days + 1 day prediction)
    split_date = df.index[-31]
    X_train, X_test = features[:split_date], features[split_date:]
    y_train, y_test = target[:split_date], target[split_date:]

    # Ensure X_train and X_test are pandas DataFrames with feature names
    X_train = pd.DataFrame(X_train, columns=features.columns)
    X_test = pd.DataFrame(X_test, columns=features.columns)

    # Step 2: Train Random Forest Model
    model_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model_rf.fit(X_train, y_train)

    # Step 3: Make Predictions
    y_pred = model_rf.predict(X_test)

    # Step 4: Estimate Confidence Intervals using Prediction Variance
    # Get predictions from each tree in the forest
    predictions_per_tree = np.array([tree.predict(X_test) for tree in model_rf.estimators_])

    # Calculate mean and percentiles across trees
    mean_predictions = predictions_per_tree.mean(axis=0)
    lower_bound = np.percentile(predictions_per_tree, 5, axis=0)  # 5th percentile for 90% CI
    upper_bound = np.percentile(predictions_per_tree, 95, axis=0)  # 95th percentile for 90% CI
    
    # SANITY CHECK: Ensure predictions have the same length as test data
    assert len(mean_predictions) == len(y_test), "Error: Length mismatch between predicted and actual values."
    assert len(lower_bound) == len(y_test), "Error: Length mismatch between lower bound predictions and actual values."
    assert len(upper_bound) == len(y_test), "Error: Length mismatch between upper bound predictions and actual values."

    # Step 5: Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # SANITY CHECK: Ensure feature importance DataFrame is not empty
    assert not feature_importance.empty, "Error: Feature importance DataFrame is empty."

    print("Feature Importance:")
    print(feature_importance)

    # Step 6: Plot Results using Plotly
    fig1 = go.Figure()

    # Actual Price
    fig1.add_trace(go.Scatter(
        x=y_test.index[:-1],
        y=y_test[:-1],
        mode='lines+markers',
        name='Actual Price',
        marker=dict(color='blue')
    ))

    # Predicted Price
    fig1.add_trace(go.Scatter(
        x=y_test.index[:-1],
        y=mean_predictions[:-1],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(dash='dash'),
        marker=dict(color='orange')
    ))

    # Confidence Interval
    fig1.add_trace(go.Scatter(
        x=y_test.index[:-1].tolist() + y_test.index[:-1].tolist()[::-1],
        y=np.concatenate([lower_bound[:-1], upper_bound[:-1][::-1]]),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% CI'
    ))

    # Next Day Prediction
    last_date = y_test.index[-1] + pd.Timedelta(days=1)
    fig1.add_trace(go.Scatter(
        x=[last_date],
        y=[mean_predictions[-1]],
        mode='markers',
        name=f'Next Day Prediction ({mean_predictions[-1]:.2f})',
        marker=dict(color='red', size=10)
    ))

    fig1.update_layout(
        title=f'Tesla Stock High Price Prediction with Random Forest<br>{dataset_name}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.01),
        template='plotly_white'
    )
    fig1.show()

    # Feature Importance Plot
    top_10_features = feature_importance.head(10)
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        y=top_10_features['Feature'],
        x=top_10_features['Importance'],
        orientation='h',
        marker=dict(color='skyblue', line=dict(color='black', width=1)),
        name='Feature Importance'
    ))

    fig2.update_layout(
        title=f'Top 10 Most Important Features<br>{dataset_name}',
        xaxis_title='Feature Importance Score',
        yaxis_title='Features',
        yaxis=dict(autorange="reversed"),
        template='plotly_white'
    )
    fig2.show()

    # Step 7: Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test[:-1], mean_predictions[:-1]))
    print(f"RMSE on Test Set: {rmse:.2f}")

    return {
        'rmse': rmse,
        'predictions': pd.DataFrame({
            'Actual': y_test,
            'Predicted': mean_predictions,
            'Lower_CI': lower_bound,
            'Upper_CI': upper_bound
        }, index=X_test.index),
        'next_day_prediction': mean_predictions[-1],
        'feature_importance': feature_importance
    }



#SEGMENT: Analyzing Data with Only the Base Features : Open, High, Low, Close, Volume
print("Analyzing Data with Only the Base Features : Open, High, Low, Close, Volume\n")
base = random_forest_regression(df_base, "Base Data")
print(base)
print("\n", "#"*100, "\n")

#SEGMENT: Analyzing Data with Base Features + Technical Indicators : RSI, MACD, etc
print("Analyzing Data with Base Features + Technical Indicators : RSI, MACD, etc\n")
technical = random_forest_regression(df_technical, "Base Data + Technical Indicators")
print(technical)
print("\n", "#"*100, "\n")

#SEGMENT: Analyzing Data with Base Features + Technical Indicators + Sentiment Analysis: Reddit, Guardian, NYT
print("Analyzing Data with Base Features + Technical Indicators + Sentiment Analysis: Reddit, Guardian, NYT\n")
sentiment = random_forest_regression(df, "Base Data + Technical Indicators + Sentiment Analysis")
print(sentiment)
print("\n", "#"*100, "\n")