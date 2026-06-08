import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/revenue_at_risk.csv")

df = load_data()

st.title("Bank Customer Churn Intelligence Dashboard")

st.write(
    "This dashboard analyses bank customer churn patterns, identifies high-risk customers, "
    "estimates revenue at risk, and supports data-driven retention decisions."
)

# Sidebar filters
st.sidebar.header("Filters")

selected_geography = st.sidebar.multiselect(
    "Geography",
    options=sorted(df["Geography"].unique()),
    default=sorted(df["Geography"].unique())
)

selected_risk = st.sidebar.multiselect(
    "Risk Category",
    options=sorted(df["RiskCategory"].unique()),
    default=sorted(df["RiskCategory"].unique())
)

selected_cluster = st.sidebar.multiselect(
    "Customer Cluster",
    options=sorted(df["Cluster"].unique()),
    default=sorted(df["Cluster"].unique())
)

filtered_df = df[
    (df["Geography"].isin(selected_geography)) &
    (df["RiskCategory"].isin(selected_risk)) &
    (df["Cluster"].isin(selected_cluster))
]

# Executive overview
st.subheader("Executive Overview")

total_customers = len(filtered_df)
actual_churn_rate = filtered_df["Exited"].mean() * 100
avg_churn_probability = filtered_df["ChurnProbability"].mean() * 100
high_risk_customers = (filtered_df["RiskCategory"] == "High Risk").sum()
total_revenue_at_risk = filtered_df["RevenueAtRisk"].sum()
avg_revenue_at_risk = filtered_df["RevenueAtRisk"].mean()

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Actual Churn Rate", f"{actual_churn_rate:.2f}%")
col3.metric("Avg Churn Probability", f"{avg_churn_probability:.2f}%")
col4.metric("High Risk Customers", f"{high_risk_customers:,}")
col5.metric("Total Revenue at Risk", f"${total_revenue_at_risk:,.0f}")
col6.metric("Avg Revenue at Risk", f"${avg_revenue_at_risk:,.0f}")

st.divider()

# Risk tier summary
st.subheader("Risk Tier Summary")

tier_summary = (
    filtered_df.groupby("RiskCategory")
    .agg(
        CustomerCount=("RevenueAtRisk", "count"),
        TotalRevenueAtRisk=("RevenueAtRisk", "sum"),
        AvgRevenueAtRisk=("RevenueAtRisk", "mean"),
        AvgChurnProbability=("ChurnProbability", "mean")
    )
    .reset_index()
)

tier_summary["RevenueAtRiskShare"] = (
    tier_summary["TotalRevenueAtRisk"] / tier_summary["TotalRevenueAtRisk"].sum() * 100
)

tier_summary = tier_summary.sort_values(
    by="TotalRevenueAtRisk",
    ascending=False
)

st.dataframe(
    tier_summary.style.format({
        "TotalRevenueAtRisk": "${:,.0f}",
        "AvgRevenueAtRisk": "${:,.0f}",
        "AvgChurnProbability": "{:.2%}",
        "RevenueAtRiskShare": "{:.2f}%"
    }),
    use_container_width=True
)

st.divider()

# Charts row 1
col_left, col_right = st.columns(2)

with col_left:
    churn_counts = filtered_df["Exited"].value_counts().reset_index()
    churn_counts.columns = ["Exited", "Count"]
    churn_counts["Customer Status"] = churn_counts["Exited"].map({
        0: "Retained",
        1: "Churned"
    })

    fig = px.pie(
        churn_counts,
        names="Customer Status",
        values="Count",
        title="Customer Churn Distribution",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    risk_summary = (
        filtered_df.groupby("RiskCategory")["RevenueAtRisk"]
        .sum()
        .reset_index()
        .sort_values(by="RevenueAtRisk", ascending=False)
    )

    fig = px.bar(
        risk_summary,
        x="RiskCategory",
        y="RevenueAtRisk",
        title="Revenue at Risk by Risk Category",
        text_auto=",.0f"
    )

    fig.update_layout(
        xaxis_title="Risk Category",
        yaxis_title="Revenue at Risk"
    )

    st.plotly_chart(fig, use_container_width=True)

# Charts row 2
col_left, col_right = st.columns(2)

with col_left:
    geo_summary = (
        filtered_df.groupby("Geography")["RevenueAtRisk"]
        .sum()
        .reset_index()
        .sort_values(by="RevenueAtRisk", ascending=False)
    )

    fig = px.bar(
        geo_summary,
        x="Geography",
        y="RevenueAtRisk",
        title="Revenue at Risk by Geography",
        text_auto=",.0f"
    )

    fig.update_layout(
        xaxis_title="Geography",
        yaxis_title="Revenue at Risk"
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    cluster_summary = (
        filtered_df.groupby("Cluster")["RevenueAtRisk"]
        .sum()
        .reset_index()
        .sort_values(by="RevenueAtRisk", ascending=False)
    )

    fig = px.bar(
        cluster_summary,
        x="Cluster",
        y="RevenueAtRisk",
        title="Revenue at Risk by Customer Cluster",
        text_auto=",.0f"
    )

    fig.update_layout(
        xaxis_title="Customer Cluster",
        yaxis_title="Revenue at Risk"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Top customers
st.subheader("Top High-Value At-Risk Customers")

top_customers = (
    filtered_df.sort_values(by="RevenueAtRisk", ascending=False)
    [[
        "Geography",
        "Gender",
        "Age",
        "Balance",
        "EstimatedSalary",
        "ChurnProbability",
        "RiskCategory",
        "Cluster",
        "EstimatedCustomerValue",
        "RevenueAtRisk"
    ]]
    .head(10)
)

st.dataframe(
    top_customers.style.format({
        "Balance": "${:,.0f}",
        "EstimatedSalary": "${:,.0f}",
        "ChurnProbability": "{:.2%}",
        "EstimatedCustomerValue": "${:,.0f}",
        "RevenueAtRisk": "${:,.0f}"
    }),
    use_container_width=True
)