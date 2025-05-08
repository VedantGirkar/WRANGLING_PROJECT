import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import plotly.graph_objects as go

df = pd.read_csv('final_data.csv')

df = df.fillna(0)
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].shift(1)

df_base = df.filter(items=['Date', 'High', 'Low', 'Close', 'Volume', "Open"])
df_technical = df.drop(columns=['reddit_sentiment', 'guardian_sentiment', 'reddit_score', 'nyt_sentiment'])

def xgboost_regressor(df):
    # Preprocess data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Create target variable (next day's High)
    df['Target'] = df['High'].shift(-1)
    df.dropna(subset=['Target'], inplace=True)

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

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test[:-1], y_pred_mean[:-1]))

    # Feature importance
    importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model_mean.feature_importances_
    }).sort_values(by='Importance', ascending=False)

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
        title='Apple Stock High Price Prediction with XGBoost',
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
        title='Top 10 Most Important Features',
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

# Example usage:
results = xgboost_regressor(df)
print(results)