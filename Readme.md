📈 Financial Market Analysis & Stock Prediction
🎯 Objective

This project analyzes historical stock market data to identify price trends, measure volatility and risk, study inter-stock correlations, and build a basic predictive model for future price movements using machine learning techniques.

The workflow covers data acquisition, technical indicator computation, exploratory analysis, and predictive modeling.

🗂 Project Structure
financial-market-analysis/
│
├── data/
│   ├── raw/                # downloaded raw CSV data
│   └── processed/          # cleaned & feature-engineered data
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_technical_indicators.ipynb
│   ├── 03_analysis_visualization.ipynb
│   └── 04_prediction_model.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── features.py
│   └── model.py
│
├── outputs/
│   ├── charts/
│   └── model_results/
│
├── main.py
└── README.md

📊 Dataset Source

Historical stock price data is fetched from:

Stooq Financial Database
Used as a reliable alternative to Yahoo Finance when API limits or network blocks occur.

Stocks analyzed:

AAPL — Apple Inc.

TSLA — Tesla Inc.

S&P 500 Index

Data frequency: Daily OHLCV
Time range: User-controlled (typically 2010 → present)

🧰 Tools & Libraries

Python 3.x

Pandas — data manipulation

NumPy — numerical computation

Scikit-learn — machine learning models

Plotly — interactive time series charts

Matplotlib — basic plotting

TA indicators implemented manually (no TA-Lib dependency required)

🔍 Key Analyses Performed
✅ Time Series Analysis

Closing price trends

Moving averages (20, 50, 200 day)

Rolling statistics

✅ Technical Indicators

SMA / EMA

RSI

MACD

Bollinger Bands

Daily Returns

✅ Risk & Volatility Metrics

Rolling volatility

Return distribution

Drawdown behavior

✅ Correlation Analysis

Cross-stock correlation matrix

Heatmap visualization

Market dependency insights

✅ Feature Engineering

Lag features

Rolling statistics

Indicator-based signals

🤖 Predictive Modeling

Two model options supported:

Linear Regression (Baseline)

Predict next-day closing price

Feature-based regression

Easy interpretability

LSTM (Optional Advanced)

Sequence-based prediction

Deep learning time series model

Better temporal pattern capture

📈 Visualizations

Interactive charts built with Plotly:

Price vs Moving Averages

Indicator overlays

Volatility curves

Correlation heatmaps

Predicted vs Actual price plots

📏 Model Evaluation Metrics

Models are evaluated using:

MAE — Mean Absolute Error

RMSE — Root Mean Square Error

R² Score

Prediction vs Actual comparison plots

▶️ How to Run
Install dependencies
pip install pandas numpy scikit-learn plotly matplotlib

Run main pipeline
python main.py

Or use notebooks

Open Jupyter:

jupyter notebook


Run notebooks in order:

01 → data loading
02 → indicators
03 → analysis
04 → prediction

⚠️ Notes

Stooq provides full history — date filtering is applied after download

Early rows of indicators may contain NaN due to lookback windows (expected)

Predictive models are educational and not financial advice

Results are sensitive to feature choice and date range

🚀 Possible Extensions

Add portfolio optimization

Add backtesting engine

Deploy dashboard (Streamlit / Dash)

Add multi-asset prediction

Integrate live data feeds

Hyperparameter tuning

📌 Academic Use

This project demonstrates:

Financial time series handling

Technical indicator computation

Quantitative risk analysis

ML-based forecasting

Reproducible data science workflow
