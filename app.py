import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

BASE = Path(__file__).resolve().parent
sales_model = pickle.load(open(BASE / "models" / "sales_rf.pkl", "rb"))
customers_model = pickle.load(open(BASE / "models" / "customers_rf.pkl", "rb"))

st.set_page_config(page_title="Rossmann Sales Forecasting", page_icon="📈", layout="wide")
st.title("📈 Rossmann Store Sales Forecasting")
st.caption("Machine Learning inference dashboard — NextHikes IT Solutions project")

def make_features(df):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    d["Day"] = d["Date"].dt.day
    d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["IsWeekend"] = d["DayOfWeek"].isin([6, 7]).astype(int)
    d["MonthPart"] = pd.cut(d["Day"], [0,10,20,31], labels=["Begin","Mid","End"]).astype(str)
    d["CompetitionOpenMonths"] = ((d["Year"]-d["CompetitionOpenSinceYear"])*12 +
                                  (d["Month"]-d["CompetitionOpenSinceMonth"])).clip(lower=0).fillna(0)
    d["Promo2OpenWeeks"] = ((d["Year"]-d["Promo2SinceYear"])*52 +
                            (d["WeekOfYear"]-d["Promo2SinceWeek"])).clip(lower=0).fillna(0)
    return d

def predict(df):
    d = make_features(df)
    for c in ["StoreType","Assortment","StateHoliday","MonthPart"]:
        d[c] = d[c].fillna("Unknown").astype(str)
    d["CompetitionDistance"] = d["CompetitionDistance"].fillna(d["CompetitionDistance"].median() if d["CompetitionDistance"].notna().any() else 0)
    sales = np.maximum(sales_model.predict(d), 0)
    customers = np.maximum(customers_model.predict(d), 0)
    return sales, customers

st.sidebar.header("Prediction input")
store_id = st.sidebar.number_input("Store ID", min_value=1, value=1, step=1)
date = st.sidebar.date_input("Date")
day_of_week = st.sidebar.selectbox("Day of Week", list(range(1,8)), index=0)
promo = st.sidebar.selectbox("Promo", [0,1], index=0)
school_holiday = st.sidebar.selectbox("School Holiday", [0,1], index=0)
store_type = st.sidebar.selectbox("Store Type", ["a","b","c","d"])
assortment = st.sidebar.selectbox("Assortment", ["a","b","c"])
competition_distance = st.sidebar.number_input("Competition Distance (m)", min_value=0.0, value=500.0)
promo2 = st.sidebar.selectbox("Promo2", [0,1], index=0)

uploaded = st.file_uploader("Upload prediction CSV (optional)", type=["csv"])
required = ["Date","DayOfWeek","Promo","SchoolHoliday","StoreType","Assortment","CompetitionDistance","Promo2",
            "CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2SinceWeek","Promo2SinceYear"]

if uploaded:
    df = pd.read_csv(uploaded)
    if "Store" not in df.columns:
        df["Store"] = store_id
    st.write("Uploaded data preview", df.head())
else:
    df = pd.DataFrame([{
        "Store": store_id, "Date": str(date), "DayOfWeek": day_of_week, "Promo": promo,
        "SchoolHoliday": school_holiday, "StoreType": store_type, "Assortment": assortment,
        "CompetitionDistance": competition_distance, "CompetitionOpenSinceMonth": 1,
        "CompetitionOpenSinceYear": 2013, "Promo2": promo2, "Promo2SinceWeek": 1,
        "Promo2SinceYear": 2013, "StateHoliday": "0"
    }])

# Fill optional fields expected by the model
defaults = {
    "StateHoliday":"0","CompetitionOpenSinceMonth":1,"CompetitionOpenSinceYear":2013,
    "Promo2SinceWeek":1,"Promo2SinceYear":2013
}
for c,v in defaults.items():
    if c not in df.columns: df[c] = v

if st.button("Predict Sales & Customers", type="primary"):
    try:
        sales, customers = predict(df)
        result = df[["Store","Date"]].copy()
        result["Predicted_Sales"] = np.round(sales,2)
        result["Predicted_Customers"] = np.round(customers,0).astype(int)
        c1,c2 = st.columns(2)
        c1.metric("Predicted Sales", f"{sales[0]:,.0f}")
        c2.metric("Predicted Customers", f"{customers[0]:,.0f}")
        st.subheader("Prediction table")
        st.dataframe(result, use_container_width=True)
        chart = result.set_index("Date")[["Predicted_Sales","Predicted_Customers"]]
        st.line_chart(chart)
        st.download_button("⬇️ Download Predictions CSV",
                           result.to_csv(index=False).encode("utf-8"),
                           "rossmann_predictions.csv","text/csv")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("For uploaded files, keep the feature names compatible with the project model.")
