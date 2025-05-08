import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import plotly.graph_objects as go

df = pd.read_csv('final_data.csv')

# SANITY CHECK: Ensure the DataFrame is not empty
assert not df.empty, "Error: The loaded DataFrame is empty."

df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].shift(1)
df = df.fillna(0)
# SANITY CHECK: Verify no missing values after shifting
assert df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().sum() == 0, "Error: Missing values found after shifting."


df_base = df.filter(items=['Date', 'High', 'Low', 'Close', 'Volume', "Open"])
# SANITY CHECK: Ensure subsets contain the expected columns
expected_base_columns = {'Date', 'High', 'Low', 'Close', 'Volume', 'Open'}
missing_base_columns = expected_base_columns - set(df_base.columns)
assert not missing_base_columns, f"Error: Missing expected columns in df_base: {missing_base_columns}"

df_technical = df.drop(columns=['reddit_sentiment', 'guardian_sentiment', 'reddit_score', 'nyt_sentiment'])
# SANITY CHECK: Ensure subsets contain the expected columns
expected_technical_columns = set(df.columns) - {'reddit_sentiment', 'guardian_sentiment', 'reddit_score', 'nyt_sentiment'}
missing_technical_columns = expected_technical_columns - set(df_technical.columns)
assert not missing_technical_columns, f"Error: Missing expected columns in df_technical: {missing_technical_columns}"

def xgboost_regressor(df, dataset_name):
    # Preprocess data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # SANITY CHECK: Ensure 'Date' column is successfully converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(df.index), "Error: 'Date' column is not properly converted to datetime."

    # Create target variable (next day's High)
    df['Target'] = df['High'].shift(-1)
    df.dropna(subset=['Target'], inplace=True)

    # SANITY CHECK: Ensure 'Target' column exists and has no missing values
    assert 'Target' in df.columns, "Error: 'Target' column not found in the DataFrame."
    assert df['Target'].isnull().sum() == 0, "Error: Missing values found in 'Target' column."
    
    # Features and target
    features = df.drop(columns=['High', 'Target'])
    target = df['Target']

    # Split into train and test (last 30 days + 1 day prediction)
    split_date = df.index[-31]
    X_train, X_test = features[:split_date], features[split_date:]
    y_train, y_test = target[:split_date], target[split_date:]

    # Train mean prediction model
    model_mean = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=5.0,  # Increase L2 regularization
        reg_alpha=2.0,  # Add L1 regularization
        gamma=0.5,
        random_state=42
    )
    model_mean.fit(X_train, y_train)

    # Train quantile models for 95% CI
    base_params = model_mean.get_params()
    base_params.pop('objective', None)  # Remove default objective

    model_lower = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=0.1,  # Correct parameter name
        **base_params
    )
    model_upper = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=0.90,  # Correct parameter name
        **base_params
    )

    model_lower.fit(X_train, y_train)
    model_upper.fit(X_train, y_train)

    # Predictions
    y_pred_mean = model_mean.predict(X_test)
    y_pred_lower = model_lower.predict(X_test)
    y_pred_upper = model_upper.predict(X_test)
    
    # SANITY CHECK: Ensure predictions have the same length as test data
    assert len(y_pred_mean) == len(y_test), "Error: Length mismatch between predicted and actual values."
    assert len(y_pred_lower) == len(y_test), "Error: Length mismatch between lower bound predictions and actual values."
    assert len(y_pred_upper) == len(y_test), "Error: Length mismatch between upper bound predictions and actual values."

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test[:-1], y_pred_mean[:-1]))

    # Feature importance
    importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model_mean.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # SANITY CHECK: Ensure feature importance DataFrame is not empty
    assert not importance.empty, "Error: Feature importance DataFrame is empty."

    # Plot results using Plotly
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
        y=y_pred_mean[:-1],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(dash='dash'),
        marker=dict(color='orange')
    ))

    # Confidence Interval
    fig1.add_trace(go.Scatter(
        x=y_test.index[:-1].tolist() + y_test.index[:-1].tolist()[::-1],
        y=np.concatenate([y_pred_lower[:-1], y_pred_upper[:-1][::-1]]),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% CI'
    ))

    # Next Day Prediction
    last_date = y_test.index[-1] + pd.Timedelta(days=1)
    fig1.add_trace(go.Scatter(
        x=[last_date],
        y=[y_pred_mean[-1]],
        mode='markers',
        name=f'Next Day Prediction ({y_pred_mean[-1]:.2f})',
        marker=dict(color='red', size=10)
    ))

    fig1.update_layout(
        title=f'Apple Stock High Price Prediction with XGBoost<br>{dataset_name}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.01),
        template='plotly_white'
    )
    fig1.show()

    # Feature Importance Plot
    top_10_features = importance.head(10)
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

    return {
        'rmse': rmse,
        'feature_importance': importance,
        'predictions': pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred_mean,
            'Lower_CI': y_pred_lower,
            'Upper_CI': y_pred_upper
        }, index=X_test.index),
        'next_day_prediction': y_pred_mean[-1]
    }

#SEGMENT: Analyzing Data with Only the Base Features : Open, High, Low, Close, Volume
print("Analyzing Data with Only the Base Features : Open, High, Low, Close, Volume\n")
base = xgboost_regressor(df_base, "Base Data")
print(base)
print("\n", "#"*100, "\n")

#SEGMENT: Analyzing Data with Base Features + Technical Indicators : RSI, MACD, etc
print("Analyzing Data with Base Features + Technical Indicators : RSI, MACD, etc\n")
technical = xgboost_regressor(df_technical, "Base Data + Technical Indicators")
print(technical)
print("\n", "#"*100, "\n")

#SEGMENT: Analyzing Data with Base Features + Technical Indicators + Sentiment Analysis: Reddit, Guardian, NYT
print("Analyzing Data with Base Features + Technical Indicators + Sentiment Analysis: Reddit, Guardian, NYT\n")
sentiment = xgboost_regressor(df, "Base Data + Technical Indicators + Sentiment Analysis")
print(sentiment)
print("\n", "#"*100, "\n")