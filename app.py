import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    layout="wide"
)

st.title("Bank Customer Churn Intelligence Dashboard")

st.write(
    "This dashboard analyses bank customer churn patterns, identifies high-risk customers, "
    "and supports data-driven retention decisions."
)

@st.cache_data
def load_data():
    return pd.read_csv("data/bank_customer_churn.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churn_rate = df["Exited"].mean() * 100
active_customers = df["IsActiveMember"].sum()
avg_balance = df["Balance"].mean()

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Active Customers", f"{active_customers:,}")
col4.metric("Avg Balance", f"${avg_balance:,.2f}")