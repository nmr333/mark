# stock_analyzer.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import ta

# -----------------------
# إعدادات عامة
# -----------------------
FUTURE_DAYS = 7  # توقع سعر الإغلاق بعد N يوم
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# -----------------------
# دوال مساعدة لجلب البيانات
# -----------------------
def fetch_price_history(ticker, period="3y", interval="1d"):
    """
    يجلب تاريخ السعر لسهم باستخدام yfinance
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, actions=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    df = df.rename(columns=lambda s: s.lower())
    df.index = pd.to_datetime(df.index)
    return df

def fetch_fundamentals(ticker):
    """
    يحاول جلب بعض القيم الأساسية المتاحة من yfinance.Ticker.info
    ملاحظة: ليست كل الحقول متاحة لكل سهم.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    # قائمة جزئية من الفلاتر المطلوبة — يمكنك توسيعها بحسب الحاجة
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
        "longBusinessSummary": info.get("longBusinessSummary"),
        "latestEarningsDate": info.get("earningsTimestamp"),
        "recommendationMean": info.get("recommendationMean")
    }
    return fields

# -----------------------
# مؤشرات فنية
# -----------------------
def add_technical_indicators(df):
    """ يستقبل DataFrame يحتوي عمود 'close' و 'volume' ويضيف أعمدة فنية"""
    out = df.copy()
    # Simple Moving Averages
    out['sma20'] = out['close'].rolling(20).mean()
    out['sma50'] = out['close'].rolling(50).mean()
    out['sma200'] = out['close'].rolling(200).mean()
    # RSI 14
    out['rsi14'] = ta.momentum.rsi(out['close'], window=14)
    # Volatility (20-day std of returns)
    out['returns'] = out['close'].pct_change()
    out['volatility20'] = out['returns'].rolling(20).std() * np.sqrt(252)  # annualized
    # 20-day high/low ranges
    out['20d_high'] = out['high'].rolling(20).max()
    out['20d_low'] = out['low'].rolling(20).min()
    out['52w_high'] = out['high'].rolling(252).max()
    out['52w_low'] = out['low'].rolling(252).min()
    # Relative Strength to SMA50: ratio
    out['close_sma50_ratio'] = out['close'] / out['sma50']
    # Average Volume (20-day)
    out['avg_volume20'] = out['volume'].rolling(20).mean()
    # Drop helper
    out = out.drop(columns=['returns'])
    return out

# -----------------------
# تحضير ميزات وهدف
# -----------------------
def prepare_features(df, fundamentals_dict, future_days=FUTURE_DAYS):
    """
    يعيد X, y حيث y هو سعر الإغلاق بعد future_days
    يضيف قيم الأساسيات كميزات ثابتة لكل صف (ستنساب مع التواريخ)
    """
    df2 = df.copy().sort_index()
    df2['future_close'] = df2['close'].shift(-future_days)
    # صفوف بدون هدف لا نحتاجها
    df2 = df2.dropna(subset=['future_close'])
    # إدراج بعض الحقول الأساسية (ثابتة عبر التاريخ) — إذا لم توجد تُستخدم NaN
    for k, v in fundamentals_dict.items():
        df2[f'fund_{k}'] = v
    # اختيار أعمدة الميزات (يمكن تعديل/إضافة)
    feature_cols = [
        'open','high','low','close','volume',
        'sma20','sma50','sma200','rsi14','volatility20',
        '20d_high','20d_low','52w_high','52w_low',
        'close_sma50_ratio','avg_volume20'
    ] + [c for c in df2.columns if c.startswith('fund_')]
    # التخلص من أي أعمدة مفقودة بالكامل
    feature_cols = [c for c in feature_cols if c in df2.columns]
    X = df2[feature_cols].copy()
    y = df2['future_close'].copy()
    # ملء القيم الفارغة بطريقة بسيطة (يمكن تحسينها)
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return X, y

# -----------------------
# تدريب نموذج وتوقع
# -----------------------
def train_and_evaluate(X, y, model_type='xgboost'):
    """
    تدريب نموذج للتنبؤ بالسعر المستقبلي.
    يعيد النموذج، ومقاييس الأداء.
    """
    # تقسيم زمنياً: نخزن آخر جزء للاختبار
    split_index = int(len(X) * (1 - TRAIN_TEST_SPLIT_RATIO))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Pipeline
    scaler = StandardScaler()
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, tree_method='auto')
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    pipeline = Pipeline([('scaler', scaler), ('model', model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return pipeline, {'rmse': rmse, 'r2': r2, 'n_test': len(y_test)}

# -----------------------
# تحليل سهم واحد (كبيرة)
# -----------------------
def analyze_ticker(ticker, period="3y", future_days=FUTURE_DAYS, model_type='xgboost'):
    print(f"Analyzing {ticker} ...")
    price_df = fetch_price_history(ticker, period=period)
    fundamentals = fetch_fundamentals(ticker)
    price_with_tech = add_technical_indicators(price_df)
    X, y = prepare_features(price_with_tech, fundamentals, future_days=future_days)
    if len(X) < 100:
        print("Warning: not enough data points to train reliably.")
    model, metrics = train_and_evaluate(X, y, model_type=model_type)
    print("Metrics:", metrics)

    # آخر صف متاح للتنبؤ المقبل
    last_row = X.iloc[-1:]
    predicted = model.predict(last_row)[0]

    # تجميع ملخّص
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

# -----------------------
# مثال تنفيذي: تحليل قائمة من الأسهم
# -----------------------
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA"]  # غيّر القائمة حسب ما تريد
    results = []
    for t in tickers:
        try:
            summary, model, X, y = analyze_ticker(t, period="5y", future_days=FUTURE_DAYS, model_type='xgboost')
            results.append(summary)
        except Exception as e:
            print(f"Error analyzing {t}: {e}")

    df_results = pd.DataFrame(results)
    print(df_results)
    df_results.to_csv("stock_analysis_summary.csv", index=False)
    print("Saved stock_analysis_summary.csv")
