import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from utils import render_sidebar, render_topnav, base_layout, COMMON_CSS
from utils import GOLD, GOLD2, RED, AMBER, STONE, TEXT, CLUSTER_LABELS

st.set_page_config(page_title="Prediction Centre", page_icon="🔮", layout="wide")
st.markdown(COMMON_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/churn_predictions.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/churn_prediction_model.joblib")

df    = load_data()
model = load_model()

render_sidebar()
render_topnav("Prediction Centre")

st.markdown('<div class="page-title">Prediction Centre</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Input a customer profile · Live churn probability from the XGBoost model · Updates instantly</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin:12px 0 20px 0;'>", unsafe_allow_html=True)

# layout
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        age          = st.slider("Age", 18, 92, 42)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        tenure       = st.slider("Tenure (years)", 0, 10, 5)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    with r1c2:
        balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
        salary  = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=80000.0, step=1000.0)
        points  = st.slider("Points Earned", 100, 1000, 500)
        sat_score = st.selectbox("Satisfaction Score (1–5)", [1, 2, 3, 4, 5], index=2)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1: geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    with r2c2: gender    = st.selectbox("Gender", ["Female", "Male"])
    with r2c3: card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])

    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1: has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    with r3c2: is_active   = st.selectbox("Active Member", ["Yes", "No"])
    with r3c3: cluster     = st.selectbox("Customer Segment", list(CLUSTER_LABELS.values()))

# build input & run model
cluster_num = [k for k, v in CLUSTER_LABELS.items() if v == cluster][0]
input_data  = pd.DataFrame([{
    "CreditScore":       credit_score,
    "Geography":         geography,
    "Gender":            gender,
    "Age":               age,
    "Tenure":            tenure,
    "Balance":           balance,
    "NumOfProducts":     num_products,
    "HasCrCard":         1 if has_cr_card == "Yes" else 0,
    "IsActiveMember":    1 if is_active   == "Yes" else 0,
    "EstimatedSalary":   salary,
    "SatisfactionScore": sat_score,
    "CardType":          card_type,
    "PointEarned":       points,
    "Cluster":           cluster_num,
}])

churn_prob  = model.predict_proba(input_data)[0][1]
churn_pred  = model.predict(input_data)[0]
risk_label  = "High Risk"   if churn_prob >= 0.70 else "Medium Risk" if churn_prob >= 0.30 else "Low Risk"
risk_color  = RED           if risk_label == "High Risk" else AMBER if risk_label == "Medium Risk" else "#16a34a"

# right panel: results
with right:
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

    # gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        number=dict(suffix="%", font=dict(color=risk_color, size=42, family="Inter")),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor=STONE, tickfont=dict(color=TEXT, size=10)),
            bar=dict(color=risk_color, thickness=0.25),
            bgcolor="#292524", borderwidth=0,
            steps=[
                dict(range=[0,  30], color="#1c1917"),
                dict(range=[30, 70], color="#292524"),
                dict(range=[70,100], color="#3d1515"),
            ],
            threshold=dict(line=dict(color=risk_color, width=3), thickness=0.75, value=churn_prob*100),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#1c1917", font=dict(family="Inter", color=TEXT),
        margin=dict(t=20, b=10, l=20, r=20), height=270,
    )
    st.plotly_chart(fig, use_container_width=True)

    # risk badge
    st.markdown(
        f"<div style='text-align:center;margin:-10px 0 16px 0;'>"
        f"<span style='background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}66;"
        f"font-size:1rem;font-weight:700;padding:6px 28px;border-radius:6px;letter-spacing:0.05em;'>"
        f"{risk_label}</span></div>",
        unsafe_allow_html=True,
    )

    # stat cards
    s1, s2, s3 = st.columns(3)
    for col, label, val, c in [
        (s1, "Churn Probability", f"{churn_prob*100:.1f}%",              risk_color),
        (s2, "Predicted Outcome", "Will Churn" if churn_pred else "Will Stay", RED if churn_pred else "#16a34a"),
        (s3, "Risk Category",     risk_label,                            risk_color),
    ]:
        with col:
            st.markdown(
                f"<div style='background:#292524;border:1px solid #44403c;border-radius:8px;"
                f"padding:12px;text-align:center;'>"
                f"<div style='font-size:0.65rem;color:#78716c;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:4px;'>{label}</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{c};'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # drivers
    st.markdown('<div class="section-header">What\'s Driving This?</div>', unsafe_allow_html=True)

    drivers = []
    if age > 55:
        drivers.append(("Age",       f"{age} yrs",   "Older customers churn significantly more",         RED))
    if num_products >= 3:
        drivers.append(("Products",  str(num_products), "3+ products is strongly linked to high churn",  RED))
    if is_active == "No":
        drivers.append(("Activity",  "Inactive",     "Inactive members churn at 2× the rate",            RED))
    if geography == "Germany":
        drivers.append(("Geography", "Germany",      "Highest churn region — 32%+ churn rate",           AMBER))
    if balance == 0:
        drivers.append(("Balance",   "$0",           "Zero balance linked to elevated churn",             AMBER))
    if gender == "Female":
        drivers.append(("Gender",    "Female",       "Female customers churn slightly more",              AMBER))
    if tenure <= 2:
        drivers.append(("Tenure",    f"{tenure} yrs","New customers carry higher churn risk",             AMBER))
    if num_products == 2 and is_active == "Yes":
        drivers.append(("Products",  "2 + Active",   "Optimal combo — associated with lowest churn",     "#16a34a"))
    if geography != "Germany" and age < 40:
        drivers.append(("Profile",   "Young + non-DE","Lower-risk demographic profile",                  "#16a34a"))
    if credit_score >= 750:
        drivers.append(("Credit",    f"{credit_score}","High credit score — lower financial stress risk","#16a34a"))

    if drivers:
        for feat, val, reason, color in drivers[:5]:
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;"
                f"background:#292524;border:1px solid #44403c;border-left:3px solid {color};"
                f"border-radius:6px;padding:10px 12px;'>"
                f"<div style='min-width:90px;font-size:0.7rem;font-weight:600;color:{color};"
                f"text-transform:uppercase;letter-spacing:0.06em;padding-top:1px;'>"
                f"{feat}<br><span style='font-size:0.85rem;'>{val}</span></div>"
                f"<div style='font-size:0.78rem;color:#a8a29e;line-height:1.5;'>{reason}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="insight-box">✓ No major risk flags detected for this customer profile.</div>', unsafe_allow_html=True)

# similar customers
st.markdown('<div class="section-header">Similar Customers in Dataset</div>', unsafe_allow_html=True)

age_range = (age - 5, age + 5)
similar = df[
    (df["Age"].between(*age_range)) &
    (df["Geography"]    == geography) &
    (df["NumOfProducts"] == num_products)
].copy()

if len(similar) > 0:
    sim_churn = similar["Exited"].mean() * 100

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in [
        (m1, "Similar Customers Found", f"{len(similar):,}"),
        (m2, "Their Actual Churn Rate",  f"{sim_churn:.1f}%"),
        (m3, "Age Range Filter",         f"{age_range[0]}–{age_range[1]} yrs"),
        (m4, "Filters Applied",          f"{geography} · {num_products} products"),
    ]:
        with col:
            st.markdown(
                f"<div style='background:#292524;border:1px solid #44403c;border-radius:8px;"
                f"padding:12px;text-align:center;'>"
                f"<div style='font-size:0.65rem;color:#78716c;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:4px;'>{label}</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{GOLD2};'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure(go.Histogram(
        x=similar["ChurnProbability"], nbinsx=25,
        marker=dict(color=GOLD, opacity=0.8, line=dict(color="#1c1917", width=0.5)),
        hovertemplate="Prob %{x:.2f}: %{y} customers<extra></extra>",
    ))
    fig.add_vline(
        x=churn_prob, line_dash="solid", line_color=risk_color, line_width=2,
        annotation_text=f"This customer ({churn_prob*100:.0f}%)",
        annotation_font_color=risk_color, annotation_font_size=11,
    )
    l = base_layout("Churn Probability Distribution — Similar Customers")
    l["xaxis"]["title"] = "Churn Probability"
    l["yaxis"]["title"] = "Count"
    l["height"]  = 280
    l["margin"]  = dict(t=40, b=32, l=8, r=20)
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

    # comparison insight
    diff = churn_prob*100 - sim_churn
    direction = "above" if diff > 0 else "below"
    diff_color = RED if diff > 5 else "#16a34a" if diff < -5 else AMBER
    st.markdown(
        f'<div class="insight-box">⚡ This customer\'s predicted churn probability is '
        f'<b style="color:{diff_color};">{abs(diff):.1f}pp {direction}</b> the average for similar customers '
        f'({sim_churn:.1f}% actual churn rate among {len(similar):,} comparable profiles).</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="insight-box">No closely matching customers found with current filters.</div>', unsafe_allow_html=True)

st.markdown("<hr style='margin:32px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='display:flex;justify-content:space-between;'>"
    "<span style='font-size:0.72rem;color:#44403c;'>Bank Churn Intelligence · Prediction Centre</span>"
    "<span style='font-size:0.72rem;color:#44403c;'>XGBoost · Scikit-learn · Streamlit · Plotly</span>"
    "</div>", unsafe_allow_html=True,
)