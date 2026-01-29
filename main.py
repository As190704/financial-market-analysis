import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import plotly.graph_objects as go
import plotly.express as px


# -------------------------
# CONFIG
# -------------------------

STOCKS = ["AAPL", "TSLA", "^GSPC"]
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"

DATA_RAW_DIR = "data/raw"
DATA_PROC_DIR = "data/processed"
REPORT_DIR = "reports/charts"
MODEL_DIR = "models"

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROC_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------
# DATA DOWNLOAD
# -------------------------

def download_data():
    print("Downloading stock data...")
    data = yf.download(STOCKS, start=START_DATE, end=END_DATE)

    close = data["Close"].dropna()

    for col in close.columns:
        close[[col]].to_csv(f"{DATA_RAW_DIR}/{col}_raw.csv")

    close.to_csv(f"{DATA_PROC_DIR}/merged_prices.csv")
    return close


# -------------------------
# FEATURE ENGINEERING
# -------------------------

def build_features(price_df, target="AAPL"):
    print("Building features...")

    df = price_df[[target]].copy()
    df["return"] = df[target].pct_change()

    df["lag1"] = df["return"].shift(1)
    df["lag2"] = df["return"].shift(2)

    df["ma10"] = df[target].rolling(10).mean()
    df["ma50"] = df[target].rolling(50).mean()

    df = df.dropna()
    df.to_csv(f"{DATA_PROC_DIR}/features.csv")

    return df


# -------------------------
# VOLATILITY + CORRELATION
# -------------------------

def risk_analysis(price_df):
    print("Running risk analysis...")

    returns = price_df.pct_change().dropna()
    returns.to_csv(f"{DATA_PROC_DIR}/returns.csv")

    vol = returns.std() * np.sqrt(252)
    print("\nAnnualized Volatility:\n", vol)

    corr = returns.corr()

    fig = px.imshow(corr, text_auto=True, title="Return Correlation")
    fig.write_html(f"{REPORT_DIR}/correlation.html")

    return returns


# -------------------------
# CHARTS
# -------------------------

def export_price_chart(price_df):
    fig = go.Figure()

    for c in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[c], name=c))

    fig.update_layout(title="Stock Price Trends")
    fig.write_html(f"{REPORT_DIR}/price_trends.html")


def export_ma_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["AAPL"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma10"], name="MA10"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma50"], name="MA50"))

    fig.update_layout(title="Moving Averages")
    fig.write_html(f"{REPORT_DIR}/moving_averages.html")


# -------------------------
# MODEL
# -------------------------

def train_model(df):
    print("Training model...")

    X = df[["lag1", "lag2", "ma10", "ma50"]]
    y = df["AAPL"]

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\nModel Metrics")
    print("RMSE:", rmse)
    print("R2:", r2)

    joblib.dump(model, f"{MODEL_DIR}/linear_regression.pkl")

    # prediction chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="Actual"))
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, name="Predicted"))
    fig.update_layout(title="Prediction vs Actual")

    fig.write_html(f"{REPORT_DIR}/predictions.html")

    return model


# -------------------------
# MAIN PIPELINE
# -------------------------

def main():
    prices = download_data()

    export_price_chart(prices)

    risk_analysis(prices)

    feature_df = build_features(prices, "AAPL")

    export_ma_chart(feature_df)

    train_model(feature_df)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
