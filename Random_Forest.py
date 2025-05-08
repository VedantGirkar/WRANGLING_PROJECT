import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv('final_data.csv')

df = df.fillna(0)
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].shift(1)

df_base = df.filter(items=['Date', 'High', 'Low', 'Close', 'Volume', "Open"])
df_technical = df.drop(columns=['reddit_sentiment', 'guardian_sentiment', 'reddit_score', 'nyt_sentiment'])


def random_forest_regression(df):
    # Step 1: Data Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

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


    # Step 7: Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test[:-1], mean_predictions[:-1]))
    print(f"RMSE on Test Set: {rmse:.2f}")

    return {
        'rmse': rmse,
        'predictions': pd.DataFrame({
            'Actual': y_test,
            'Predicted': mean_predictions
        }, index=X_test.index),
        'next_day_prediction': mean_predictions[-1]
    }


# Example Usage
# Assuming `df` is your preloaded DataFrame
results = random_forest_regression(df)
print(results)