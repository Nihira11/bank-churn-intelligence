import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Churn Analysis",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/revenue_at_risk.csv")

df = load_data()

st.title("Churn Analysis")

st.write(
    "This page explores churn patterns across customer demographics, geography, "
    "activity status, product usage, and balance groups."
)

# helper function
def churn_rate_chart(data, group_col, title):
    summary = (
        data.groupby(group_col)["Exited"]
        .mean()
        .reset_index()
    )

    summary["ChurnRate"] = summary["Exited"] * 100
    summary = summary.sort_values(by="ChurnRate", ascending=False)

    fig = px.bar(
        summary,
        x=group_col,
        y="ChurnRate",
        title=title,
        text="ChurnRate"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title="Churn Rate (%)"
    )

    return fig


# KPIs
st.subheader("Churn Overview")

total_customers = len(df)
churned_customers = df["Exited"].sum()
retained_customers = total_customers - churned_customers
churn_rate = df["Exited"].mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Retained Customers", f"{retained_customers:,}")
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# charts
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        churn_rate_chart(df, "Geography", "Churn Rate by Geography"),
        use_container_width=True
    )

with col_right:
    st.plotly_chart(
        churn_rate_chart(df, "Gender", "Churn Rate by Gender"),
        use_container_width=True
    )

col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        churn_rate_chart(df, "AgeGroup", "Churn Rate by Age Group"),
        use_container_width=True
    )

with col_right:
    activity_df = df.copy()
    activity_df["ActivityStatus"] = activity_df["IsActiveMember"].map({
        0: "Inactive",
        1: "Active"
    })

    st.plotly_chart(
        churn_rate_chart(activity_df, "ActivityStatus", "Churn Rate by Activity Status"),
        use_container_width=True
    )

col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        churn_rate_chart(df, "NumOfProducts", "Churn Rate by Number of Products"),
        use_container_width=True
    )

with col_right:
    st.plotly_chart(
        churn_rate_chart(df, "BalanceGroup", "Churn Rate by Balance Group"),
        use_container_width=True
    )

st.divider()

st.subheader("Key Churn Insights")

st.markdown(
    """
    - Germany shows the highest churn rate among the available geographies.
    - Inactive customers are more likely to churn than active customers.
    - Customers with three or four products show unusually high churn rates, though the four-product group is small.
    - Balance group behaviour is not perfectly linear, suggesting churn is influenced by multiple factors rather than balance alone.
    """
)