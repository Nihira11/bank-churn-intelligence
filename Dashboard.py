import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import render_sidebar, render_topnav, base_layout, COMMON_CSS
from utils import GOLD, GOLD2, RED, AMBER, STONE, TEXT, CLUSTER_LABELS

st.set_page_config(page_title="Bank Churn Intelligence", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")
st.markdown(COMMON_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/churn_predictions.csv")
    if "RevenueAtRisk" not in df.columns:
        df["BalanceScore"] = df["Balance"] / df["Balance"].max()
        df["SalaryScore"]  = df["EstimatedSalary"] / df["EstimatedSalary"].max()
        df["TenureScore"]  = df["Tenure"] / df["Tenure"].max()
        df["PointsScore"]  = df["PointEarned"] / df["PointEarned"].max()
        df["CustomerValueScore"] = 0.40*df["BalanceScore"] + 0.30*df["SalaryScore"] + 0.20*df["TenureScore"] + 0.10*df["PointsScore"]
        df["EstimatedCustomerValue"] = df["CustomerValueScore"] * df["EstimatedSalary"]
        df["RevenueAtRisk"] = df["EstimatedCustomerValue"] * df["ChurnProbability"]
    return df

df_full = load_data()

# sidebar filters
def filters():
    global sel_geo, sel_risk
    geo_options  = ["All Regions"] + sorted(df_full["Geography"].unique().tolist())
    risk_options = ["All Risk Tiers", "High Risk", "Medium Risk", "Low Risk"]
    sel_geo  = st.selectbox("Geography", geo_options,  label_visibility="collapsed", key="dash_geo")
    sel_risk = st.selectbox("Risk Tier",  risk_options, label_visibility="collapsed", key="dash_risk")
    df_f = df_full.copy()
    if sel_geo  != "All Regions":    df_f = df_f[df_f["Geography"]    == sel_geo]
    if sel_risk != "All Risk Tiers": df_f = df_f[df_f["RiskCategory"] == sel_risk]
    st.markdown(f"<div style='font-size:0.75rem;color:#57534e;margin-top:12px;'>{len(df_f):,} customers selected</div>", unsafe_allow_html=True)
    return df_f

sel_geo = "All Regions"; sel_risk = "All Risk Tiers"
render_sidebar()

# apply filters manually after sidebar
df = df_full.copy()
with st.sidebar:
    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#57534e;margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)
    sel_geo  = st.selectbox("Geography",  ["All Regions"] + sorted(df_full["Geography"].unique().tolist()),  label_visibility="collapsed", key="dash_geo")
    sel_risk = st.selectbox("Risk Tier",  ["All Risk Tiers","High Risk","Medium Risk","Low Risk"], label_visibility="collapsed", key="dash_risk")
    if sel_geo  != "All Regions":    df = df[df["Geography"]    == sel_geo]
    if sel_risk != "All Risk Tiers": df = df[df["RiskCategory"] == sel_risk]
    st.markdown(f"<div style='font-size:0.75rem;color:#57534e;margin-top:12px;'>{len(df):,} customers selected</div>", unsafe_allow_html=True)

# top nav
render_topnav("Overview")

# header
col_title, col_badge = st.columns([4,1])
with col_title:
    st.markdown('<div class="page-title">Bank Churn Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Customer retention analytics · Predictive churn scoring · Revenue risk monitoring</div>', unsafe_allow_html=True)
with col_badge:
    st.markdown("<div style='text-align:right;padding-top:6px;'><span style='background:#292524;border:1px solid #44403c;color:#a8a29e;font-size:0.7rem;padding:4px 10px;border-radius:4px;'>LIVE DEMO</span></div>", unsafe_allow_html=True)
st.markdown("<hr style='margin:12px 0 20px 0;'>", unsafe_allow_html=True)

# metrics
total_customers    = len(df)
churn_rate         = df["Exited"].mean() * 100
predicted_churners = int(df["PredictedChurn"].sum())
high_risk_count    = int((df["RiskCategory"] == "High Risk").sum())
total_rar          = df["RevenueAtRisk"].sum()
avg_rar            = df["RevenueAtRisk"].mean()
high_risk_rar_pct  = (df[df["RiskCategory"]=="High Risk"]["RevenueAtRisk"].sum() / total_rar * 100) if total_rar > 0 else 0

k1,k2,k3,k4,k5 = st.columns(5)
for col, label, value, sub, badge in [
    (k1,"Total Customers",    f"{total_customers:,}",   "dataset scope",     "10K records"),
    (k2,"Historical Churn",   f"{churn_rate:.1f}%",     "actual churn rate", "1 in 5 customers"),
    (k3,"Predicted to Churn", f"{predicted_churners:,}","model prediction",  "XGBoost · ROC-AUC 0.88"),
    (k4,"High-Risk Customers",f"{high_risk_count:,}",   "churn prob ≥ 70%",  f"{high_risk_rar_pct:.0f}% of revenue risk"),
    (k5,"Revenue at Risk",    f"${total_rar/1e6:.1f}M", "estimated exposure",f"avg ${avg_rar:,.0f} / customer"),
]:
    with col:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div><div class="kpi-badge">{badge}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1 ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Churn Overview</div>', unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)

with c1:
    counts = df["Exited"].value_counts().reset_index()
    counts.columns = ["Status","Count"]
    counts["Status"] = counts["Status"].map({0:"Retained",1:"Churned"})
    fig = go.Figure(go.Pie(
        labels=counts["Status"], values=counts["Count"], hole=0.65,
        marker=dict(colors=["#44403c",GOLD], line=dict(color="#1c1917",width=2)),
        textinfo="percent", textfont=dict(color="#fef3c7",size=12),
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:10px'>churn</span>",
                       x=0.5,y=0.5,showarrow=False,font=dict(color=GOLD2,size=16))
    l = base_layout("Retained vs Churned")
    l["showlegend"]=True
    l["legend"]=dict(font=dict(color=TEXT),bgcolor="rgba(0,0,0,0)",x=0.5,xanchor="center",y=-0.08,orientation="h")
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    risk_order = ["High Risk","Medium Risk","Low Risk"]
    rc = df["RiskCategory"].value_counts().reindex(risk_order).reset_index()
    rc.columns = ["RiskCategory","Count"]
    fig = go.Figure()
    for _, row in rc.iterrows():
        c = {"High Risk":RED,"Medium Risk":AMBER,"Low Risk":GOLD}.get(row["RiskCategory"],GOLD)
        fig.add_trace(go.Bar(x=[row["RiskCategory"]],y=[row["Count"]],name=row["RiskCategory"],
                             marker_color=c,text=[f"{int(row['Count']):,}"],textposition="outside",
                             textfont=dict(color="#e7e5e4",size=11),cliponaxis=False,
                             hovertemplate="%{x}: %{y:,}<extra></extra>"))
    l = base_layout("Customers by Risk Tier"); l["showlegend"]=False; l["yaxis"]["title"]="Customers"
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    geo_rar = df.groupby("Geography")["RevenueAtRisk"].sum().reset_index().sort_values("RevenueAtRisk",ascending=True)
    geo_rar["RAR_M"] = geo_rar["RevenueAtRisk"]/1e6
    x_max = geo_rar["RAR_M"].max()*1.3
    fig = go.Figure(go.Bar(
        x=geo_rar["RAR_M"],y=geo_rar["Geography"],orientation="h",
        marker=dict(color=geo_rar["RAR_M"],colorscale=[[0,"#44403c"],[1,GOLD]],showscale=False),
        text=geo_rar["RAR_M"].apply(lambda x:f"${x:.1f}M"),textposition="outside",
        textfont=dict(color="#e7e5e4",size=11),cliponaxis=False,
        hovertemplate="%{y}: $%{x:.1f}M<extra></extra>",
    ))
    l = base_layout("Revenue at Risk by Country"); l["xaxis"]["title"]="$M"; l["xaxis"]["range"]=[0,x_max]
    l["yaxis"]["showgrid"]=False; l["margin"]=dict(t=40,b=32,l=8,r=90)
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2 ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
c4,c5 = st.columns([1.3,1])

with c4:
    fig = go.Figure(go.Histogram(
        x=df["ChurnProbability"],nbinsx=40,
        marker=dict(color=GOLD,opacity=0.85,line=dict(color="#1c1917",width=0.5)),
        hovertemplate="Prob %{x:.2f}: %{y:,} customers<extra></extra>",
    ))
    fig.add_vline(x=0.30,line_dash="dot",line_color=AMBER,line_width=1.5,annotation_text="Medium",annotation_font_color=AMBER,annotation_font_size=10)
    fig.add_vline(x=0.70,line_dash="dot",line_color=RED,line_width=1.5,annotation_text="High",annotation_font_color=RED,annotation_font_size=10)
    l = base_layout("Churn Probability Distribution"); l["xaxis"]["title"]="Churn Probability"; l["yaxis"]["title"]="Customers"
    l["bargap"]=0.04; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

with c5:
    df["Persona"] = df["Cluster"].map(CLUSTER_LABELS)
    from utils import CLUSTER_COLORS
    persona_rar = df.groupby("Persona")["RevenueAtRisk"].sum().reset_index().sort_values("RevenueAtRisk",ascending=True)
    persona_rar["RAR_M"] = persona_rar["RevenueAtRisk"]/1e6
    x_max2 = persona_rar["RAR_M"].max()*1.3
    fig = go.Figure(go.Bar(
        x=persona_rar["RAR_M"],y=persona_rar["Persona"],orientation="h",
        marker=dict(color=persona_rar["RAR_M"],colorscale=[[0,"#44403c"],[1,GOLD]],showscale=False),
        text=persona_rar["RAR_M"].apply(lambda x:f"${x:.1f}M"),textposition="outside",
        textfont=dict(color="#e7e5e4",size=11),cliponaxis=False,
        hovertemplate="%{y}: $%{x:.1f}M<extra></extra>",
    ))
    l = base_layout("Revenue at Risk by Customer Persona"); l["xaxis"]["title"]="$M"; l["xaxis"]["range"]=[0,x_max2]
    l["yaxis"]["showgrid"]=False; l["margin"]=dict(t=40,b=32,l=8,r=90)
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3 ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Churn Drivers</div>', unsafe_allow_html=True)
c6,c7,c8 = st.columns(3)

with c6:
    a = df.groupby("AgeGroup")["Exited"].mean().reset_index(); a["pct"]=a["Exited"]*100
    fig = go.Figure(go.Bar(x=a["AgeGroup"].astype(str),y=a["pct"],marker_color=GOLD,opacity=0.9,
        text=a["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=10),cliponaxis=False))
    l = base_layout("Churn Rate by Age Group"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c7:
    g = df.groupby("Geography")["Exited"].mean().reset_index().sort_values("Exited",ascending=False); g["pct"]=g["Exited"]*100
    fig = go.Figure(go.Bar(x=g["Geography"],y=g["pct"],marker_color=[RED,AMBER,GOLD][:len(g)],
        text=g["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=10),cliponaxis=False))
    l = base_layout("Churn Rate by Geography"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c8:
    p = df.groupby("NumOfProducts")["Exited"].mean().reset_index(); p["pct"]=p["Exited"]*100
    fig = go.Figure(go.Bar(x=p["NumOfProducts"].astype(str),y=p["pct"],marker_color=GOLD,opacity=0.9,
        text=p["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=10),cliponaxis=False))
    l = base_layout("Churn Rate by No. of Products"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr style='margin:32px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;'><span style='font-size:0.72rem;color:#44403c;'>Bank Churn Intelligence · Portfolio Project</span><span style='font-size:0.72rem;color:#44403c;'>XGBoost · Scikit-learn · Streamlit · Plotly</span></div>", unsafe_allow_html=True)