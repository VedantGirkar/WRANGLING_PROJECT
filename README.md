# Market Mood Swing Analysis Project

## Overview
This project aims to analyze market mood swings by predicting tesla stock price movements and leveraging sentiment 
analysis from external sources like Reddit, The Guardian, and the New York Times. Using machine learning models such 
as Random Forest and XGBoost, we forecast the next day's Close price of Tesla stock while estimating confidence 
intervals and identifying key features influencing predictions.

The project provides interactive visualizations using **Plotly** to enhance the interpretability of results, allowing users to explore trends, confidence intervals, and feature importance dynamically.

---

## Table of Contents
1. [Objective](#objective)
2. [Data Sources](#data-sources)
3. [Methodology](#methodology)
4. [Code Structure](#code-structure)
5. [Results](#results)
6. [Interactive Visualizations](#interactive-visualizations)
7. [Future Improvements](#future-improvements)

---

## Order of Operations
1. NYC.py: Fetches data from the New York Times API and saves it as a CSV file.
2. Data_Ingest.py: Ingests the data for the Stocks, Reddit Sentiment, and Guardian Sentiment, & combines them into a 
   single DataFrame with the NYC data.
3. Data Analysis : There are two models, Random Forest and XGBoost, which are trained on the combined DataFrame.
3.1 Random Forest: Trains a Random Forest Regressor to predict the next day's Close price.
3.2 XGBoost: Trains an XGBoost Regressor with quantile regression to estimate confidence intervals.

---

## Objective
The primary goal of this project is to:
- Predict the next day's high price of Tesla stock.
- Estimate confidence intervals for predictions to quantify uncertainty.
- Identify the most important features driving stock price predictions.
- Analyze how external sentiment (e.g., Reddit, news articles) impacts stock prices.

---

## Data Sources
The project utilizes the following datasets:
1. **Stock Price Data**: Historical data including Open, High, Low, Close, and Volume for Tesla stock.
2. **Sentiment Data**:
   - **Reddit Sentiment**: Aggregated sentiment scores from Reddit posts related to Tesla.
   - **News Sentiment**: Sentiment scores from articles in The Guardian and the New York Times.
3. **Technical Indicators**: Derived features such as moving averages, RSI, MACD, etc., computed from stock price data.

All datasets are preprocessed to handle missing values and ensure compatibility with machine learning models.

---

## Methodology
### 1. Data Preprocessing
- Convert the `Date` column to a datetime format and set it as the index.
- Create a target variable (`Target`) representing the next day's Close price.
- Shift features to avoid data leakage and drop rows with missing values.

### 2. Feature Engineering
- Extract relevant features from stock price data and sentiment sources.

### 3. Model Selection
Two models are implemented:
- **Random Forest Regressor**: To capture non-linear relationships and provide feature importance.
- **XGBoost Regressor**: For enhanced performance with regularization and quantile regression to estimate confidence intervals.

### 4. Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Measures the prediction accuracy.
- Confidence intervals (90%) are estimated using quantile regression (XGBoost) or prediction variance (Random Forest).

### 5. Visualization
Interactive plots are created using **Plotly** to display:
- Actual vs. predicted stock prices with confidence intervals.
- Feature importance rankings.

---

## Code Structure
The project consists of five main components:

### 1. Data Ingestion
- Fetch historical stock price data using `yfinance`.
- Collect sentiment data from Reddit, The Guardian, and the New York Times APIs.
- Merge all data into a single DataFrame for analysis.

### 2. Feature Engineering
- Calculate technical indicators such as RSI, MACD, Bollinger Bands, and VWAP.
- Aggregate sentiment scores from Reddit, The Guardian, and NYT.

### 3. Model Training
- Train a **Random Forest Regressor** to predict stock prices and estimate confidence intervals using prediction variance.
- Train an **XGBoost Regressor** with quantile regression for tighter confidence intervals.

### 4. Interactive Visualizations
- Use **Plotly** to create interactive plots for:
  - Actual vs. predicted prices with confidence intervals.
  - Feature importance rankings.

### 5. Results and Analysis
- Evaluate model performance using RMSE.
- Analyze the impact of sentiment and technical indicators on stock price predictions.

---

## Results
### Key Findings
1. **Prediction Accuracy**:
   - Both Random Forest and XGBoost models achieve low RMSE values, indicating accurate predictions.
   - XGBoost provides tighter confidence intervals due to quantile regression.

2. **Feature Importance**:
   - Technical indicators (e.g., moving averages, RSI) dominate feature importance rankings.
   - Sentiment features (e.g., Reddit, news sentiment) contribute moderately to predictions.

3. **Market Mood Swings**:
   - External sentiment often correlates with significant price movements, validating its inclusion in the model.

## Interactive Visualizations

## 1. Actual vs. Predicted Prices
An interactive plot displays:

- **Actual stock prices** (blue line).
- **Predicted prices** (orange dashed line).
- **90% confidence intervals** (gray shaded area).
- **Next day's prediction** (red marker).

## 2. Feature Importance
A horizontal bar chart ranks the top 10 features by importance, highlighting their contribution to the model.

---

# Future Improvements

1. **Incorporate More Sentiment Sources**:
   - Include Twitter sentiment and other financial news platforms for a broader perspective.

2. **Advanced Models**:
   - Experiment with deep learning models (e.g., LSTM, Transformer-based models) for time-series forecasting.

3. **Real-Time Predictions**:
   - Develop a pipeline to fetch live data and make real-time predictions.

4. **Portfolio Optimization**:
   - Extend the model to predict multiple stocks and optimize investment portfolios.

5. **Enhanced Visualization**:
   - Add dashboards using tools like Dash or Streamlit for a more user-friendly interface.

---

# Conclusion
This project successfully predicts Tesla stock's next day high price and identifies key factors influencing market mood 
swings. By combining technical indicators and sentiment analysis, the models provide actionable insights for investors. The use of interactive visualizations enhances the interpretability of results, making this project a valuable tool for stock market analysis.

---

# Acknowledgments
- **Libraries Used**: Pandas, NumPy, Scikit-learn, XGBoost, Plotly, yfinance, PRAW.
- **Datasets**: Yahoo Finance, Reddit API, The Guardian API, New York Times API.