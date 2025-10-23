import streamlit as st
from stock_model import analyze_ticker
import pandas as pd

st.set_page_config(page_title="تحليل الأسهم بالذكاء الاصطناعي", layout="wide")

st.title("📊 تطبيق تحليل الأسهم الذكي")
st.write("هذا التطبيق يجمع بين البيانات الأساسية والفنية ويتنبأ بالسعر المستقبلي للسهم.")

ticker = st.text_input("أدخل رمز السهم (مثل AAPL أو MSFT أو TSLA):", "AAPL")
future_days = st.slider("عدد الأيام المستقبلية للتنبؤ:", 3, 30, 7)
period = st.selectbox("الفترة الزمنية:", ["1y", "3y", "5y", "10y"])
model_type = st.selectbox("نوع النموذج:", ["xgboost", "random_forest"])

if st.button("ابدأ التحليل"):
    with st.spinner("⏳ جاري جلب البيانات وتحليلها..."):
        try:
            summary, model, X, y = analyze_ticker(ticker, period=period, future_days=future_days, model_type=model_type)
            st.success(f"✅ تم تحليل السهم {ticker} بنجاح")
            
            st.subheader("📋 ملخص النتائج")
            st.json(summary)

            st.subheader("📈 السعر التاريخي وتوقع المستقبل")
            last_price = summary["last_close"]
            predicted = summary[f"predicted_close_in_{future_days}_days"]
            
            st.write(f"السعر الحالي: **{last_price:.2f} USD**")
            st.write(f"التوقع بعد {future_days} يوم: **{predicted:.2f} USD**")

            chart_data = pd.DataFrame({'Price': y})
            st.line_chart(chart_data)
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")
