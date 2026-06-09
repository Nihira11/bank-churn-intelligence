import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import render_sidebar, render_topnav, base_layout, COMMON_CSS
from utils import GOLD, GOLD2, RED, AMBER, STONE, TEXT

st.set_page_config(page_title="Churn Analysis", page_icon="📊", layout="wide")
st.markdown(COMMON_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/churn_predictions.csv")

df_full = load_data()

render_sidebar()
with st.sidebar:
    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#57534e;margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)
    sel_geo = st.selectbox("Geography", ["All"] + sorted(df_full["Geography"].unique().tolist()), label_visibility="collapsed", key="ca_geo")
    sel_gen = st.selectbox("Gender",    ["All"] + sorted(df_full["Gender"].unique().tolist()),    label_visibility="collapsed", key="ca_gen")
    df = df_full.copy()
    if sel_geo != "All": df = df[df["Geography"] == sel_geo]
    if sel_gen != "All": df = df[df["Gender"]    == sel_gen]
    st.markdown(f"<div style='font-size:0.75rem;color:#57534e;margin-top:12px;'>{len(df):,} customers</div>", unsafe_allow_html=True)

render_topnav("Churn Analysis")

st.markdown('<div class="page-title">Churn Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Deep-dive into churn patterns across demographics, behaviour, and product usage</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin:12px 0 20px 0;'>", unsafe_allow_html=True)

# section 1
st.markdown('<div class="section-header">Demographic Churn Patterns</div>', unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)

with c1:
    g = df.groupby("Gender")["Exited"].mean().reset_index(); g["pct"]=g["Exited"]*100
    colors = [GOLD if v==g["pct"].max() else STONE for v in g["pct"]]
    fig = go.Figure(go.Bar(x=g["Gender"],y=g["pct"],marker_color=colors,
        text=g["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by Gender"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c2:
    a = df.groupby("AgeGroup")["Exited"].mean().reset_index(); a["pct"]=a["Exited"]*100
    colors = [GOLD if v==a["pct"].max() else STONE for v in a["pct"]]
    fig = go.Figure(go.Bar(x=a["AgeGroup"].astype(str),y=a["pct"],marker_color=colors,
        text=a["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by Age Group"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c3:
    geo = df.groupby("Geography")["Exited"].mean().reset_index().sort_values("Exited",ascending=False); geo["pct"]=geo["Exited"]*100
    fig = go.Figure(go.Bar(x=geo["Geography"],y=geo["pct"],marker_color=[RED,AMBER,GOLD][:len(geo)],
        text=geo["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by Geography"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

i1,i2,i3 = st.columns(3)
top_g = df.groupby("Gender")["Exited"].mean().idxmax(); top_g_pct = df.groupby("Gender")["Exited"].mean().max()*100
top_a = df.groupby("AgeGroup")["Exited"].mean().idxmax(); top_a_pct = df.groupby("AgeGroup")["Exited"].mean().max()*100
top_geo = df.groupby("Geography")["Exited"].mean().idxmax(); top_geo_pct = df.groupby("Geography")["Exited"].mean().max()*100
with i1: st.markdown(f'<div class="insight-box">⚡ <b>{top_g}</b> customers churn at <b>{top_g_pct:.1f}%</b> – higher than the opposite group. Gender may interact with product usage and complaint behaviour.</div>', unsafe_allow_html=True)
with i2: st.markdown(f'<div class="insight-box">⚡ The <b>{top_a}</b> age group has the highest churn at <b>{top_a_pct:.1f}%</b>. Older customers may have different financial needs or find better alternatives.</div>', unsafe_allow_html=True)
with i3: st.markdown(f'<div class="insight-box">⚡ <b>{top_geo}</b> has the highest churn at <b>{top_geo_pct:.1f}%</b>. Regional product fit or local competition may be key drivers.</div>', unsafe_allow_html=True)

# section 2
st.markdown('<div class="section-header">Behavioural Churn Patterns</div>', unsafe_allow_html=True)
c4,c5,c6 = st.columns(3)

with c4:
    act = df.groupby("IsActiveMember")["Exited"].mean().reset_index(); act["pct"]=act["Exited"]*100
    act["label"] = act["IsActiveMember"].map({0:"Inactive",1:"Active"})
    fig = go.Figure(go.Bar(x=act["label"],y=act["pct"],marker_color=[RED,GOLD],
        text=act["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by Activity Status"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c5:
    comp = df.groupby("Complain")["Exited"].mean().reset_index(); comp["pct"]=comp["Exited"]*100
    comp["label"] = comp["Complain"].map({0:"No Complaint",1:"Complained"})
    fig = go.Figure(go.Bar(x=comp["label"],y=comp["pct"],marker_color=[GOLD,RED],
        text=comp["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by Complaint Status"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c6:
    sat = df.groupby("SatisfactionScore")["Exited"].mean().reset_index(); sat["pct"]=sat["Exited"]*100
    fig = go.Figure(go.Bar(x=sat["SatisfactionScore"].astype(str),y=sat["pct"],marker_color=GOLD,opacity=0.85,
        text=sat["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by Satisfaction Score"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

i4,i5,i6 = st.columns(3)
inactive_pct = df[df["IsActiveMember"]==0]["Exited"].mean()*100; active_pct = df[df["IsActiveMember"]==1]["Exited"].mean()*100
comp_pct = df[df["Complain"]==1]["Exited"].mean()*100; no_comp_pct = df[df["Complain"]==0]["Exited"].mean()*100
sat_diff = (df.groupby("SatisfactionScore")["Exited"].mean().max() - df.groupby("SatisfactionScore")["Exited"].mean().min())*100
with i4: st.markdown(f'<div class="insight-box">⚡ Inactive customers churn at <b>{inactive_pct:.1f}%</b> vs <b>{active_pct:.1f}%</b> for active members. Re-engagement campaigns could significantly reduce attrition.</div>', unsafe_allow_html=True)
with i5: st.markdown(f'<div class="insight-box">⚡ Customers who complained churn at <b>{comp_pct:.1f}%</b> vs <b>{no_comp_pct:.1f}%</b>. Complaint resolution is the single highest-leverage retention action.</div>', unsafe_allow_html=True)
with i6: st.markdown(f'<div class="insight-box">⚡ Satisfaction score has only a <b>{sat_diff:.1f}pp</b> spread – it\'s a weak standalone predictor of churn in this dataset.</div>', unsafe_allow_html=True)

# section 3
st.markdown('<div class="section-header">Product & Balance Churn Patterns</div>', unsafe_allow_html=True)
c7,c8,c9 = st.columns(3)

with c7:
    prod = df.groupby("NumOfProducts")["Exited"].mean().reset_index(); prod["pct"]=prod["Exited"]*100
    colors = [RED if v>50 else GOLD for v in prod["pct"]]
    fig = go.Figure(go.Bar(x=prod["NumOfProducts"].astype(str),y=prod["pct"],marker_color=colors,
        text=prod["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by No. of Products"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c8:
    bal_order = ["No Balance","Low","Medium","High","Very High"]
    bal = df.groupby("BalanceGroup")["Exited"].mean().reset_index(); bal["pct"]=bal["Exited"]*100
    bal["BalanceGroup"] = pd.Categorical(bal["BalanceGroup"],categories=bal_order,ordered=True)
    bal = bal.sort_values("BalanceGroup")
    fig = go.Figure(go.Bar(x=bal["BalanceGroup"].astype(str),y=bal["pct"],marker_color=GOLD,opacity=0.85,
        text=bal["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by Balance Group"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

with c9:
    card = df.groupby("CardType")["Exited"].mean().reset_index().sort_values("Exited",ascending=False); card["pct"]=card["Exited"]*100
    fig = go.Figure(go.Bar(x=card["CardType"],y=card["pct"],marker_color=GOLD,opacity=0.85,
        text=card["pct"].apply(lambda x:f"{x:.1f}%"),textposition="outside",textfont=dict(color="#e7e5e4",size=11),cliponaxis=False))
    l = base_layout("Churn Rate by Card Type"); l["yaxis"]["title"]="Churn Rate (%)"; l["margin"]=dict(t=40,b=32,l=8,r=20)
    fig.update_layout(**l); st.plotly_chart(fig, use_container_width=True)

i7,i8,i9 = st.columns(3)
p2 = df[df["NumOfProducts"]==2]["Exited"].mean()*100; p3 = df[df["NumOfProducts"]==3]["Exited"].mean()*100
top_bal = df.groupby("BalanceGroup")["Exited"].mean().idxmax(); top_bal_pct = df.groupby("BalanceGroup")["Exited"].mean().max()*100
card_diff = (df.groupby("CardType")["Exited"].mean().max() - df.groupby("CardType")["Exited"].mean().min())*100
with i7: st.markdown(f'<div class="insight-box">⚡ 2-product customers churn at just <b>{p2:.1f}%</b>, but 3+ product customers spike to <b>{p3:.1f}%</b>+. More products don\'t mean more loyalty.</div>', unsafe_allow_html=True)
with i8: st.markdown(f'<div class="insight-box">⚡ The <b>{top_bal}</b> balance group has the highest churn at <b>{top_bal_pct:.1f}%</b>. These customers represent the most significant revenue risk.</div>', unsafe_allow_html=True)
with i9: st.markdown(f'<div class="insight-box">⚡ Card type shows only a <b>{card_diff:.1f}pp</b> churn spread – it\'s not a strong discriminator. Behavioural factors matter more.</div>', unsafe_allow_html=True)

# heatmap
st.markdown('<div class="section-header">Age × Balance Churn Heatmap</div>', unsafe_allow_html=True)
heatmap_data = df.groupby(["AgeGroup","BalanceGroup"])["Exited"].mean().unstack(fill_value=0)*100
bal_order_h = [b for b in ["No Balance","Low","Medium","High","Very High"] if b in heatmap_data.columns]
heatmap_data = heatmap_data[bal_order_h]
fig = go.Figure(go.Heatmap(
    z=heatmap_data.values, x=heatmap_data.columns.tolist(), y=heatmap_data.index.astype(str).tolist(),
    colorscale=[[0,"#1c1917"],[0.4,"#78350f"],[0.7,AMBER],[1.0,GOLD2]],
    text=[[f"{v:.1f}%" for v in row] for row in heatmap_data.values],
    texttemplate="%{text}", textfont=dict(color="#fef3c7",size=11),
    hovertemplate="Age: %{y} | Balance: %{x}<br>Churn Rate: %{z:.1f}%<extra></extra>",
    showscale=True, colorbar=dict(tickfont=dict(color=TEXT),outlinewidth=0),
))
l = base_layout("Churn Rate (%) by Age Group and Balance Group")
l["height"]=300; l["margin"]=dict(t=40,b=20,l=8,r=8)
fig.update_layout(**l)
st.plotly_chart(fig, use_container_width=True)
st.markdown('<div class="insight-box">⚡ The heatmap reveals which <b>age × balance combinations</b> carry the highest churn risk. High-balance older customers are the most financially exposed segment – primary targets for personalised retention outreach.</div>', unsafe_allow_html=True)

st.markdown("<hr style='margin:32px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;'><span style='font-size:0.72rem;color:#44403c;'>Bank Churn Intelligence · Churn Analysis</span><span style='font-size:0.72rem;color:#44403c;'>XGBoost · Scikit-learn · Streamlit · Plotly</span></div>", unsafe_allow_html=True)