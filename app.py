import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Customer Churn & Retention Analytics",
    layout="wide"
)

st.title("Customer Churn & Retention Analytics Platform")

st.write(
    "This app analyses customer churn patterns, predicts churn risk, "
    "and provides retention recommendations."
)

@st.cache_data
def load_data():
    return pd.read_csv("data/telco_churn.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)

total_customers = len(df)
churn_rate = (df["Churn"].value_counts(normalize=True)["Yes"] * 100)
avg_monthly_charges = df["MonthlyCharges"].mean()

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")