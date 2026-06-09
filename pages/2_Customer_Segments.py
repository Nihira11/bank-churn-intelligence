import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from utils import render_sidebar, render_topnav, base_layout, COMMON_CSS
from utils import GOLD, GOLD2, RED, AMBER, STONE, TEXT
from utils import CLUSTER_LABELS, CLUSTER_COLORS, CLUSTER_RISK, CLUSTER_RISK_COLORS, CLUSTER_DESC

st.set_page_config(page_title="Customer Segments", page_icon="👥", layout="wide")
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
with st.sidebar:
    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#57534e;margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)
    cluster_opts = ["All Segments"] + [CLUSTER_LABELS[i] for i in sorted(CLUSTER_LABELS)]
    sel_cluster = st.selectbox("Segment", cluster_opts, label_visibility="collapsed", key="cs_seg")
    df_view = df if sel_cluster == "All Segments" else df[df["Persona"] == sel_cluster]
    st.markdown(f"<div style='font-size:0.75rem;color:#57534e;margin-top:12px;'>{len(df_view):,} customers</div>", unsafe_allow_html=True)

render_topnav("Customer Segments")

st.markdown('<div class="page-title">Customer Segments</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">K-Means clustering · 4 behavioural personas · Churn risk by segment</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin:12px 0 20px 0;'>", unsafe_allow_html=True)

# persona cards
st.markdown('<div class="section-header">Customer Personas</div>', unsafe_allow_html=True)
cols = st.columns(4)
for i, col in enumerate(cols):
    seg = df[df["Cluster"]==i]
    with col:
        st.markdown(f"""
        <div class="persona-card" style="border-top-color:{CLUSTER_COLORS[i]};">
          <div class="persona-name">{CLUSTER_LABELS[i]}</div>
          <span class="persona-tag" style="background:{CLUSTER_RISK_COLORS[i]}22;color:{CLUSTER_RISK_COLORS[i]};border:1px solid {CLUSTER_RISK_COLORS[i]}44;">{CLUSTER_RISK[i]}</span>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:4px;">
            <div><div class="persona-stat-label">Customers</div><div class="persona-stat-value">{len(seg):,}</div></div>
            <div><div class="persona-stat-label">Churn Rate</div><div class="persona-stat-value" style="color:{CLUSTER_COLORS[i]};">{seg["Exited"].mean()*100:.1f}%</div></div>
            <div><div class="persona-stat-label">Avg Balance</div><div class="persona-stat-value">${seg["Balance"].mean():,.0f}</div></div>
            <div><div class="persona-stat-label">Avg Age</div><div class="persona-stat-value">{seg["Age"].mean():.0f} yrs</div></div>
          </div>
          <div class="persona-desc">{CLUSTER_DESC[i]}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# segment comparison
st.markdown('<div class="section-header">Segment Comparison</div>', unsafe_allow_html=True)
profile = df.groupby("Cluster").agg(
    ChurnRate=("Exited","mean"), AvgAge=("Age","mean"), AvgBalance=("Balance","mean"),
    AvgTenure=("Tenure","mean"), AvgProducts=("NumOfProducts","mean"),
    ActiveRate=("IsActiveMember","mean"), RevenueAtRisk=("RevenueAtRisk","sum"),
).reset_index()
profile["Persona"]    = profile["Cluster"].map(CLUSTER_LABELS)
profile["ChurnPct"]   = profile["ChurnRate"]*100
profile["RAR_M"]      = profile["RevenueAtRisk"]/1e6
profile["ActivePct"]  = profile["ActiveRate"]*100
colors = [CLUSTER_COLORS[i] for i in profile["Cluster"]]

c1,c2,c3 = st.columns(3)
for col, y_col, y_fmt, title, y_title in [
    (c1, "ChurnPct",   lambda v: f"{v:.1f}%",        "Churn Rate by Segment",       "Churn Rate (%)"),
    (c2, "RAR_M",      lambda v: f"${v:.1f}M",        "Revenue at Risk by Segment",  "$M"),
    (c3, "AvgBalance", lambda v: f"${v:,.0f}",         "Avg Balance by Segment",      "Avg Balance ($)"),
]:
    with col:
        fig = go.Figure(go.Bar(
            x=profile["Persona"], y=profile[y_col], marker_color=colors,
            text=profile[y_col].apply(y_fmt), textposition="outside",
            textfont=dict(color="#e7e5e4",size=11), cliponaxis=False,
        ))
        l = base_layout(title); l["yaxis"]["title"]=y_title; l["xaxis"]["tickangle"]=-20; l["margin"]=dict(t=40,b=70,l=8,r=20)
        fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

# radar
st.markdown('<div class="section-header">Segment Radar Profile</div>', unsafe_allow_html=True)
radar_features = ["AvgAge","AvgBalance","AvgTenure","AvgProducts","ActivePct","ChurnPct"]
radar_labels   = ["Age","Balance","Tenure","Products","Active %","Churn %"]
scaler = MinMaxScaler()
radar_scaled = scaler.fit_transform(profile[radar_features])
fig = go.Figure()
fig = go.Figure()
for i, (idx_row, row) in enumerate(profile.iterrows()):
    actual_vals = [
        row["AvgAge"], row["AvgBalance"], row["AvgTenure"],
        row["AvgProducts"], row["ActivePct"], row["ChurnPct"]
    ]
    actual_labels = ["Age", "Balance", "Tenure", "Products", "Active %", "Churn %"]
    actual_fmts = [
        f"{row['AvgAge']:.0f} yrs",
        f"${row['AvgBalance']:,.0f}",
        f"{row['AvgTenure']:.1f} yrs",
        f"{row['AvgProducts']:.2f}",
        f"{row['ActivePct']:.1f}%",
        f"{row['ChurnPct']:.1f}%",
    ]
    scaled_row = radar_scaled[i]
    custom = actual_fmts + [actual_fmts[0]]  # close the loop

    fig.add_trace(go.Scatterpolar(
        r=list(scaled_row) + [scaled_row[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        name=profile["Persona"].iloc[i],
        line=dict(color=colors[i], width=2),
        fillcolor=colors[i],
        opacity=0.15,
        customdata=custom,
        hovertemplate="<b>%{theta}</b>: %{customdata}<extra>" + profile["Persona"].iloc[i] + "</extra>",
    ))

fig.update_layout(
    polar=dict(
        bgcolor="#1c1917",
        radialaxis=dict(visible=True,showticklabels=False,gridcolor="#44403c",linecolor="#44403c"),
        angularaxis=dict(gridcolor="#44403c",linecolor="#44403c",color="#a8a29e",tickfont=dict(size=11,color="#a8a29e")),
    ),
    paper_bgcolor="#1c1917", font=dict(color=TEXT,family="Inter"),
    legend=dict(font=dict(color=TEXT),bgcolor="rgba(0,0,0,0)",orientation="h",x=0.5,xanchor="center",y=-0.12),
    margin=dict(t=60, b=70, l=40, r=40), height=420,
    title=dict(text="Normalised Segment Characteristics",font=dict(color="#e7e5e4",size=13),x=0,xanchor="left"),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('<div class="insight-box">⚡ The radar shows each segment\'s relative profile across 6 dimensions. <b>Older At-Risk</b> scores highest on churn – prioritise for retention. <b>Inactive High-Balance</b> has high balance but low activity – a re-engagement opportunity. <b>Active High-Balance</b> is the healthiest segment.</div>', unsafe_allow_html=True)

# churn breakdown
st.markdown('<div class="section-header">Churn Breakdown Within Segments</div>', unsafe_allow_html=True)
c4,c5 = st.columns(2)

with c4:
    seg_retained = df[df["Exited"]==0].groupby("Persona").size().reset_index(name="Retained")
    seg_churned  = df[df["Exited"]==1].groupby("Persona").size().reset_index(name="Churned")
    seg_counts   = seg_retained.merge(seg_churned, on="Persona")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Retained",x=seg_counts["Persona"],y=seg_counts["Retained"],marker_color=STONE,hovertemplate="%{x} Retained: %{y:,}<extra></extra>"))
    fig.add_trace(go.Bar(name="Churned", x=seg_counts["Persona"],y=seg_counts["Churned"], marker_color=GOLD, hovertemplate="%{x} Churned: %{y:,}<extra></extra>"))
    l = base_layout("Retained vs Churned per Segment"); l["barmode"]="stack"; l["xaxis"]["tickangle"]=-15
    l["margin"]=dict(t=40,b=70,l=8,r=20); l["showlegend"]=True
    l["legend"]=dict(font=dict(color=TEXT),bgcolor="rgba(0,0,0,0)",orientation="h",x=0,y=1.12)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c5:
    prob_seg = df.groupby("Persona")["ChurnProbability"].mean().reset_index().sort_values("ChurnProbability",ascending=True)
    bar_colors = []
    for p in prob_seg["Persona"]:
        k = [k for k,v in CLUSTER_LABELS.items() if v==p]
        bar_colors.append(CLUSTER_COLORS[k[0]] if k else GOLD)
    fig = go.Figure(go.Bar(
        x=prob_seg["ChurnProbability"], y=prob_seg["Persona"], orientation="h",
        marker_color=bar_colors,
        text=prob_seg["ChurnProbability"].apply(lambda x:f"{x:.2f}"),
        textposition="outside", textfont=dict(color="#e7e5e4",size=11), cliponaxis=False,
    ))
    l = base_layout("Avg Churn Probability by Segment"); l["xaxis"]["title"]="Avg Churn Probability"
    l["xaxis"]["range"]=[0, prob_seg["ChurnProbability"].max()*1.3]; l["yaxis"]["showgrid"]=False
    l["margin"]=dict(t=40,b=32,l=8,r=60)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr style='margin:32px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;'><span style='font-size:0.72rem;color:#44403c;'>Bank Churn Intelligence · Customer Segments</span><span style='font-size:0.72rem;color:#44403c;'>K-Means · Scikit-learn · Streamlit · Plotly</span></div>", unsafe_allow_html=True)