import streamlit as st
import pandas as pd
from stock_model import analyze_ticker
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحليل الأسهم بالذكاء الاصطناعي", layout="wide")

st.title("📊 تطبيق تحليل الأسهم الذكي")
st.write("هذا التطبيق يجمع بين البيانات الأساسية والفنية ويتنبأ بالسعر المستقبلي للسهم.")

# 🧩 إدخال المستخدم
ticker = st.text_input("أدخل رمز السهم (مثل AAPL أو MSFT أو TSLA):", "AAPL")
future_days = st.slider("عدد الأيام المستقبلية للتنبؤ:", 3, 30, 7)
period = st.selectbox("الفترة الزمنية:", ["1y", "3y", "5y", "10y"])
model_type = st.selectbox("نوع النموذج:", ["xgboost", "random_forest"])

if st.button("ابدأ التحليل"):
    with st.spinner("⏳ جاري جلب البيانات وتحليلها..."):
        try:
            summary, model, X, y = analyze_ticker(ticker, period=period, future_days=future_days, model_type=model_type)
            st.success(f"✅ تم تحليل السهم {ticker} بنجاح")
            
            # عرض النتائج
            st.subheader("📋 ملخص النتائج")
            st.json(summary)

            # رسم السعر والتوقع
            st.subheader("📈 السعر التاريخي وتوقع المستقبل")
            last_price = summary["last_close"]
            predicted = summary[f"predicted_close_in_{future_days}_days"]
            
            st.write(f"السعر الحالي: **{last_price:.2f} USD**")
            st.write(f"التوقع بعد {future_days} يوم: **{predicted:.2f} USD**")

            plt.figure(figsize=(10, 4))
            plt.plot(y.index, y.values, label="الأسعار التاريخية")
            plt.axhline(predicted, color="red", linestyle="--", label="السعر المتوقع")
            plt.legend()
            st.pyplot(plt)
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")
