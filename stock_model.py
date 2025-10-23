# stock_model.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import ta

FUTURE_DAYS = 7
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

def fetch_price_history(ticker, period="3y", interval="1d"):
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, actions=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    df = df.rename(columns=lambda s: s.lower())
    df.index = pd.to_datetime(df.index)
    return df

def fetch_fundamentals(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    fields = {
        "exchange": info.get("exchange"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio"),
        "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
        "priceToBook": info.get("priceToBook"),
        "priceToCashflow": info.get("priceToCashflow"),
        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        "dividendYield": info.get("dividendYield"),
        "sharesOutstanding": info.get("sharesOutstanding"),
        "floatShares": info.get("floatShares"),
        "shortPercentOfFloat": info.get("shortPercentOfFloat"),
        "beta": info.get("beta"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "targetMeanPrice": info.get("targetMeanPrice"),
        "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
        "bookValue": info.get("bookValue"),
        "returnOnEquity": info.get("returnOnEquity"),
        "profitMargins": info.get("profitMargins"),
        "grossMargins": info.get("grossMargins"),
        "currentRatio": info.get("currentRatio"),
        "debtToEquity": info.get("debtToEquity"),
        "recommendationMean": info.get("recommendationMean")
    }
    return fields

def add_technical_indicators(df):
    out = df.copy()
    out['sma20'] = out['close'].rolling(20).mean()
    out['sma50'] = out['close'].rolling(50).mean()
    out['sma200'] = out['close'].rolling(200).mean()
    out['rsi14'] = ta.momentum.rsi(out['close'], window=14)
    out['returns'] = out['close'].pct_change()
    out['volatility20'] = out['returns'].rolling(20).std() * np.sqrt(252)
    out['20d_high'] = out['high'].rolling(20).max()
    out['20d_low'] = out['low'].rolling(20).min()
    out['52w_high'] = out['high'].rolling(252).max()
    out['52w_low'] = out['low'].rolling(252).min()
    out['close_sma50_ratio'] = out['close'] / out['sma50']
    out['avg_volume20'] = out['volume'].rolling(20).mean()
    out = out.drop(columns=['returns'])
    return out

def prepare_features(df, fundamentals_dict, future_days=FUTURE_DAYS):
    df2 = df.copy().sort_index()
    df2['future_close'] = df2['close'].shift(-future_days)
    df2 = df2.dropna(subset=['future_close'])

    for k, v in fundamentals_dict.items():
        df2[f'fund_{k}'] = v

    # تحويل النصوص إلى أرقام
    text_cols = df2.select_dtypes(include=['object']).columns
    df2 = pd.get_dummies(df2, columns=text_cols, drop_first=True)

    feature_cols = [c for c in df2.columns if c != 'future_close']
    X = df2[feature_cols].fillna(0)
    y = df2['future_close']
    return X, y

def train_and_evaluate(X, y, model_type='xgboost'):
    split_index = int(len(X) * (1 - TRAIN_TEST_SPLIT_RATIO))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = StandardScaler()
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, tree_method='auto')
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    pipeline = Pipeline([('scaler', scaler), ('model', model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    return pipeline, {'rmse': rmse, 'r2': r2, 'n_test': len(y_test)}

def analyze_ticker(ticker, period="3y", future_days=FUTURE_DAYS, model_type='xgboost'):
    price_df = fetch_price_history(ticker, period=period)
    fundamentals = fetch_fundamentals(ticker)
    price_with_tech = add_technical_indicators(price_df)
    X, y = prepare_features(price_with_tech, fundamentals, future_days=future_days)
    if len(X) < 100:
        print("Warning: not enough data points to train reliably.")
    model, metrics = train_and_evaluate(X, y, model_type=model_type)
    last_row = X.iloc[-1:]
    predicted = model.predict(last_row)[0]

    summary = {
        "ticker": ticker,
        "last_close": price_df['close'].iloc[-1],
        "predicted_close_in_{}_days".format(future_days): float(predicted),
        "target_mean_price": fundamentals.get("targetMeanPrice"),
        "marketCap": fundamentals.get("marketCap"),
        "sector": fundamentals.get("sector"),
        "industry": fundamentals.get("industry"),
        "country": fundamentals.get("country"),
        "model_metrics": metrics
    }
    return summary, model, X, y
