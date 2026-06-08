import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Customer Segments",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/revenue_at_risk.csv")

df = load_data()

persona_map = {
    0: "Active High-Balance Customers",
    1: "Older At-Risk Customers",
    2: "Inactive High-Balance Customers",
    3: "Low-Balance Product-Heavy Customers"
}

df["Persona"] = df["Cluster"].map(persona_map)

st.title("Customer Segments")

st.write(
    "This page converts K-Means clusters into business-friendly customer personas. "
    "Each segment is profiled using churn risk, engagement, balance, product ownership, "
    "and revenue-at-risk indicators."
)

cluster_profile = (
    df.groupby(["Cluster", "Persona"])
    .agg(
        CustomerCount=("Cluster", "count"),
        AvgAge=("Age", "mean"),
        AvgBalance=("Balance", "mean"),
        AvgSalary=("EstimatedSalary", "mean"),
        AvgProducts=("NumOfProducts", "mean"),
        ActiveRate=("IsActiveMember", "mean"),
        ChurnRate=("Exited", "mean"),
        AvgChurnProbability=("ChurnProbability", "mean"),
        TotalRevenueAtRisk=("RevenueAtRisk", "sum"),
        AvgRevenueAtRisk=("RevenueAtRisk", "mean")
    )
    .reset_index()
)

# overview KPIs
st.subheader("Segment Overview")

total_segments = df["Cluster"].nunique()
largest_cluster_row = cluster_profile.sort_values("CustomerCount", ascending=False).iloc[0]
highest_churn_row = cluster_profile.sort_values("ChurnRate", ascending=False).iloc[0]
highest_revenue_row = cluster_profile.sort_values("TotalRevenueAtRisk", ascending=False).iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customer Segments", total_segments)
col2.metric(
    "Largest Segment",
    f"Cluster {int(largest_cluster_row['Cluster'])}",
    f"{int(largest_cluster_row['CustomerCount']):,} customers"
)
col3.metric(
    "Highest Churn Segment",
    f"Cluster {int(highest_churn_row['Cluster'])}",
    f"{highest_churn_row['ChurnRate']:.2%} churn"
)
col4.metric(
    "Highest Revenue Exposure",
    f"Cluster {int(highest_revenue_row['Cluster'])}",
    f"${highest_revenue_row['TotalRevenueAtRisk']:,.0f}"
)

st.divider()

# priority segment
st.subheader("Priority Segment")

priority = highest_revenue_row

st.info(
    f"""
    **Cluster {int(priority['Cluster'])} – {priority['Persona']}**

    Revenue at Risk: ${priority['TotalRevenueAtRisk']:,.0f}

    This segment contains high-balance customers who are currently inactive.
    While Cluster 1 has the highest churn rate, Cluster 2 represents the
    largest financial exposure and should be the primary target for
    reactivation campaigns.
    """
)

# persona cards
st.subheader("Segment Comparison")

persona_summary = cluster_profile[
    [
        "Cluster",
        "Persona",
        "CustomerCount",
        "ChurnRate",
        "AvgBalance",
        "TotalRevenueAtRisk"
    ]
]

st.dataframe(
    persona_summary.style.format({
        "ChurnRate":"{:.2%}",
        "AvgBalance":"${:,.0f}",
        "TotalRevenueAtRisk":"${:,.0f}"
    }),
    use_container_width=True
)

# charts
short_persona_map = {
    "Active High-Balance Customers": "Active High-Balance",
    "Older At-Risk Customers": "Older At-Risk",
    "Inactive High-Balance Customers": "Inactive High-Balance",
    "Low-Balance Product-Heavy Customers": "Product-Heavy"
}

cluster_profile["ShortPersona"] = cluster_profile["Persona"].map(short_persona_map)

st.subheader("Segment Visual Analysis")

col_left, col_right = st.columns(2)

with col_left:
    fig = px.bar(
        cluster_profile,
        x="ShortPersona",
        y="CustomerCount",
        color="ShortPersona",
        title="Customer Distribution by Segment",
        text="CustomerCount"
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    fig = px.bar(
        cluster_profile.sort_values("ChurnRate", ascending=False),
        x="ShortPersona",
        y="ChurnRate",
        title="Churn Rate by Persona",
        text="ChurnRate"
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Persona",
        yaxis_title="Churn Rate",
        xaxis_tickangle=-20
    )

    st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns(2)

with col_left:
    fig = px.bar(
        cluster_profile.sort_values("TotalRevenueAtRisk", ascending=False),
        x="Persona",
        y="TotalRevenueAtRisk",
        title="Total Revenue at Risk by Persona",
        text_auto=",.0f"
    )

    fig.update_layout(
        xaxis_title="Persona",
        yaxis_title="Total Revenue at Risk",
        xaxis_tickangle=-30
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    fig = px.bar(
        cluster_profile.sort_values("AvgRevenueAtRisk", ascending=False),
        x="ShortPersona",
        y="AvgRevenueAtRisk",
        title="Average Revenue at Risk per Customer",
        text_auto=",.0f"
    )

    fig.update_layout(
        xaxis_title="Persona",
        yaxis_title="Average Revenue at Risk",
        xaxis_tickangle=-30
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# strategy table
st.subheader("Retention Strategy by Persona")

persona_table = pd.DataFrame({
    "Cluster": [0, 1, 2, 3],
    "Persona": [
        "Active High-Balance Customers",
        "Older At-Risk Customers",
        "Inactive High-Balance Customers",
        "Low-Balance Product-Heavy Customers"
    ],
    "Primary Risk": [
        "Complacency or future disengagement",
        "High churn likelihood",
        "Inactive despite high financial value",
        "Potential product mismatch"
    ],
    "Recommended Action": [
        "Maintain engagement through loyalty rewards and premium account benefits.",
        "Prioritise personalised retention outreach and relationship manager contact.",
        "Run reactivation campaigns focused on engagement and product usage.",
        "Monitor satisfaction and review product fit."
    ]
})

st.dataframe(
    persona_table,
    use_container_width=True,
    hide_index=True
)

st.markdown(
    """
    ### Key Segment Insight

    Cluster 2 is the highest priority from a total revenue exposure perspective, while Cluster 1 is the highest priority from a churn-rate perspective. 
    
    This means the bank should use two retention strategies: broad reactivation campaigns for Cluster 2 and targeted high-touch outreach for Cluster 1.
    """
)