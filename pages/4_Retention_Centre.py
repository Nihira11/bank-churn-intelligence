import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import render_sidebar, render_topnav, base_layout, COMMON_CSS
from utils import GOLD, GOLD2, RED, AMBER, STONE, TEXT, CLUSTER_LABELS

st.set_page_config(page_title="Retention Centre", page_icon="🎯", layout="wide")
st.markdown(COMMON_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/churn_predictions.csv")
    if "RevenueAtRisk" not in df.columns:
        df["BalanceScore"] = df["Balance"] / df["Balance"].max()
        df["SalaryScore"]  = df["EstimatedSalary"] / df["EstimatedSalary"].max()
        df["TenureScore"]  = df["Tenure"] / df["Tenure"].max()
        df["PointsScore"]  = df["PointEarned"] / df["PointEarned"].max()
        df["CustomerValueScore"] = 0.40*df["BalanceScore"]+0.30*df["SalaryScore"]+0.20*df["TenureScore"]+0.10*df["PointsScore"]
        df["EstimatedCustomerValue"] = df["CustomerValueScore"] * df["EstimatedSalary"]
        df["RevenueAtRisk"] = df["EstimatedCustomerValue"] * df["ChurnProbability"]
    return df

df = load_data()
df["Persona"] = df["Cluster"].map(CLUSTER_LABELS)

render_sidebar()
render_topnav("Retention Centre")

st.markdown('<div class="page-title">Retention Centre</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">High-risk customer identification · Revenue recovery opportunities · Retention recommendations</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin:12px 0 20px 0;'>", unsafe_allow_html=True)

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#57534e;margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)
    geo_opts  = ["All"] + sorted(df["Geography"].unique().tolist())
    seg_opts  = ["All"] + list(CLUSTER_LABELS.values())
    sel_geo   = st.selectbox("Geography", geo_opts, label_visibility="collapsed", key="rc_geo")
    sel_seg   = st.selectbox("Segment",   seg_opts, label_visibility="collapsed", key="rc_seg")
    min_prob  = st.slider("Min Churn Probability", 0.0, 1.0, 0.70, 0.05)
    top_n     = st.slider("Show Top N Customers", 10, 100, 25, 5)

    df_f = df.copy()
    if sel_geo != "All": df_f = df_f[df_f["Geography"] == sel_geo]
    if sel_seg != "All": df_f = df_f[df_f["Persona"]   == sel_seg]
    df_f = df_f[df_f["ChurnProbability"] >= min_prob]
    st.markdown(f"<div style='font-size:0.75rem;color:#57534e;margin-top:12px;'>{len(df_f):,} at-risk customers</div>", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
high_risk   = df[df["RiskCategory"] == "High Risk"]
total_rar   = df_f["RevenueAtRisk"].sum()
avg_prob    = df_f["ChurnProbability"].mean() * 100 if len(df_f) > 0 else 0
avg_balance = df_f["Balance"].mean() if len(df_f) > 0 else 0

k1, k2, k3, k4 = st.columns(4)
for col, label, value, sub in [
    (k1, "At-Risk Customers",    f"{len(df_f):,}",          f"prob ≥ {min_prob:.0%}"),
    (k2, "Revenue at Risk",      f"${total_rar/1e6:.1f}M",  "filtered segment"),
    (k3, "Avg Churn Probability",f"{avg_prob:.1f}%",         "within filter"),
    (k4, "Avg Balance at Risk",  f"${avg_balance:,.0f}",     "per customer"),
]:
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 1: revenue recovery charts ───────────────────────────────────────
st.markdown('<div class="section-header">Revenue Exposure Overview</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    rar_geo = df_f.groupby("Geography")["RevenueAtRisk"].sum().reset_index().sort_values("RevenueAtRisk", ascending=True)
    rar_geo["RAR_M"] = rar_geo["RevenueAtRisk"] / 1e6
    x_max = rar_geo["RAR_M"].max() * 1.3 if len(rar_geo) > 0 else 1
    fig = go.Figure(go.Bar(
        x=rar_geo["RAR_M"], y=rar_geo["Geography"], orientation="h",
        marker=dict(color=rar_geo["RAR_M"], colorscale=[[0,"#44403c"],[1,RED]], showscale=False),
        text=rar_geo["RAR_M"].apply(lambda x: f"${x:.1f}M"),
        textposition="outside", textfont=dict(color="#e7e5e4", size=11), cliponaxis=False,
    ))
    l = base_layout("Revenue at Risk by Region"); l["xaxis"]["title"]="$M"; l["xaxis"]["range"]=[0,x_max]
    l["yaxis"]["showgrid"]=False; l["margin"]=dict(t=40,b=32,l=8,r=80)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c2:
    rar_seg = df_f.groupby("Persona")["RevenueAtRisk"].sum().reset_index().sort_values("RevenueAtRisk", ascending=True)
    rar_seg["RAR_M"] = rar_seg["RevenueAtRisk"] / 1e6
    x_max2 = rar_seg["RAR_M"].max() * 1.3 if len(rar_seg) > 0 else 1
    fig = go.Figure(go.Bar(
        x=rar_seg["RAR_M"], y=rar_seg["Persona"], orientation="h",
        marker=dict(color=rar_seg["RAR_M"], colorscale=[[0,"#44403c"],[1,AMBER]], showscale=False),
        text=rar_seg["RAR_M"].apply(lambda x: f"${x:.1f}M"),
        textposition="outside", textfont=dict(color="#e7e5e4", size=11), cliponaxis=False,
    ))
    l = base_layout("Revenue at Risk by Segment"); l["xaxis"]["title"]="$M"; l["xaxis"]["range"]=[0,x_max2]
    l["yaxis"]["showgrid"]=False; l["margin"]=dict(t=40,b=32,l=8,r=80)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c3:
    # churn prob vs balance scatter (sample 300 for speed)
    sample = df_f.sample(min(300, len(df_f)), random_state=42) if len(df_f) > 0 else df_f
    fig = go.Figure(go.Scatter(
        x=sample["ChurnProbability"], y=sample["Balance"],
        mode="markers",
        marker=dict(
            color=sample["ChurnProbability"], colorscale=[[0,AMBER],[1,RED]],
            size=6, opacity=0.7, showscale=False,
        ),
        hovertemplate="Prob: %{x:.2f}<br>Balance: $%{y:,.0f}<extra></extra>",
    ))
    l = base_layout("Churn Probability vs Balance")
    l["xaxis"]["title"]="Churn Probability"; l["yaxis"]["title"]="Balance ($)"
    l["yaxis"]["showgrid"]=True; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

# ── Section 2: retention strategy cards ──────────────────────────────────────
st.markdown('<div class="section-header">Retention Playbook</div>', unsafe_allow_html=True)

strategies = [
    {
        "segment":  "Older At-Risk (Cluster 1)",
        "priority": "CRITICAL",
        "color":    RED,
        "icon":     "🔴",
        "actions": [
            "Assign dedicated relationship manager for personalised outreach",
            "Offer tailored retirement planning or wealth management products",
            "Proactive check-in calls before contract renewal dates",
            "Exclusive loyalty rewards for long-standing customers",
        ],
        "expected": "Target 36% churn rate → reduce to ~20% with direct engagement",
    },
    {
        "segment":  "Inactive High-Balance (Cluster 2)",
        "priority": "HIGH",
        "color":    AMBER,
        "icon":     "🟠",
        "actions": [
            "Re-engagement email/SMS campaign with product highlights",
            "Offer fee waivers or bonus interest rates for activity",
            "Personalised dashboard showing unused product benefits",
            "Targeted promotion: activate within 30 days for rewards",
        ],
        "expected": "Convert inactive → active to halve churn risk from 28% → ~14%",
    },
    {
        "segment":  "Germany Region",
        "priority": "HIGH",
        "color":    AMBER,
        "icon":     "🟠",
        "actions": [
            "Investigate local competitive landscape and pricing gaps",
            "Region-specific product bundles or fee structures",
            "German-language customer service and communications",
            "Local partnership programs or community banking initiatives",
        ],
        "expected": "Germany churn at 32% vs 16% in France/Spain – regional strategy needed",
    },
    {
        "segment":  "Complaint Resolution",
        "priority": "URGENT",
        "color":    RED,
        "icon":     "🔴",
        "actions": [
            "Same-day complaint acknowledgement SLA",
            "Escalation path to senior staff for high-value customers",
            "Post-resolution follow-up survey and retention offer",
            "Root cause analysis to fix systemic complaint drivers",
        ],
        "expected": "Complained customers churn at 99.5% – resolution is the #1 lever",
    },
]

s1, s2 = st.columns(2)
for i, strat in enumerate(strategies):
    col = s1 if i % 2 == 0 else s2
    with col:
        actions_html = "".join([
            f"<li style='margin-bottom:5px;color:#a8a29e;font-size:0.78rem;'>{a}</li>"
            for a in strat["actions"]
        ])
        st.markdown(f"""
        <div style='background:#292524;border:1px solid #44403c;border-left:4px solid {strat["color"]};
             border-radius:10px;padding:16px 18px;margin-bottom:14px;'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
            <span style='font-size:1rem;'>{strat["icon"]}</span>
            <span style='font-size:0.65rem;font-weight:700;color:{strat["color"]};text-transform:uppercase;
                  letter-spacing:0.1em;background:{strat["color"]}22;padding:2px 8px;border-radius:4px;'>
              {strat["priority"]}</span>
          </div>
          <div style='font-size:0.95rem;font-weight:700;color:#fef3c7;margin-bottom:10px;'>{strat["segment"]}</div>
          <ul style='margin:0 0 10px 16px;padding:0;'>{actions_html}</ul>
          <div style='font-size:0.72rem;color:{strat["color"]};border-top:1px solid #44403c;
               padding-top:8px;margin-top:4px;'>
            📈 {strat["expected"]}
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Section 3: top at-risk customer table ─────────────────────────────────────
st.markdown('<div class="section-header">Top At-Risk Customers</div>', unsafe_allow_html=True)

top_customers = (
    df_f.sort_values("RevenueAtRisk", ascending=False)
    .head(top_n)
    [[
        "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "IsActiveMember", "ChurnProbability",
        "RiskCategory", "RevenueAtRisk", "Persona"
    ]]
    .copy()
)

top_customers["ChurnProbability"] = (top_customers["ChurnProbability"] * 100).round(1).astype(str) + "%"
top_customers["Balance"]          = top_customers["Balance"].apply(lambda x: f"${x:,.0f}")
top_customers["RevenueAtRisk"]    = top_customers["RevenueAtRisk"].apply(lambda x: f"${x:,.0f}")
top_customers["IsActiveMember"]   = top_customers["IsActiveMember"].map({1: "Active", 0: "Inactive"})
top_customers.columns = [
    "Country", "Gender", "Age", "Tenure", "Balance",
    "Products", "Status", "Churn Prob", "Risk", "Revenue at Risk", "Segment"
]
top_customers = top_customers.reset_index(drop=True)
top_customers.index += 1

# style the dataframe
def style_risk(val):
    if val == "High Risk":   return f"color: {RED}; font-weight: 600;"
    if val == "Medium Risk": return f"color: {AMBER}; font-weight: 600;"
    return f"color: #16a34a; font-weight: 600;"

def style_status(val):
    return f"color: {RED};" if val == "Inactive" else f"color: #16a34a;"

styled = (
    top_customers.style
    .map(style_risk,    subset=["Risk"])
    .map(style_status,  subset=["Status"])
    .set_properties(**{
        "background-color": "#292524",
        "color":            "#e7e5e4",
        "border":           "1px solid #44403c",
        "font-size":        "0.8rem",
        "padding":          "6px 10px",
    })
    .set_table_styles([{
        "selector": "th",
        "props": [
            ("background-color", "#161412"),
            ("color", "#78716c"),
            ("font-size", "0.65rem"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.08em"),
            ("padding", "8px 10px"),
            ("border", "1px solid #44403c"),
        ]
    }])
)

st.dataframe(styled, use_container_width=True, height=min(400, 35 + top_n * 35))

# download button
csv = top_customers.to_csv(index=True)
st.download_button(
    label="⬇  Download At-Risk Customer List",
    data=csv,
    file_name="at_risk_customers.csv",
    mime="text/csv",
)

st.markdown("<hr style='margin:32px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='display:flex;justify-content:space-between;'>"
    "<span style='font-size:0.72rem;color:#44403c;'>Bank Churn Intelligence · Retention Centre</span>"
    "<span style='font-size:0.72rem;color:#44403c;'>XGBoost · Scikit-learn · Streamlit · Plotly</span>"
    "</div>", unsafe_allow_html=True,
)