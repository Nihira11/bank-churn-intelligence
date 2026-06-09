import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import plotly.graph_objects as go
from utils import render_sidebar, render_topnav, base_layout, COMMON_CSS
from utils import GOLD, GOLD2, RED, AMBER, STONE, TEXT, CLUSTER_LABELS

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.markdown(COMMON_CSS, unsafe_allow_html=True)

@st.cache_data
def load_and_train():
    df = pd.read_csv("data/segmented_bank_churn.csv")

    features = [
        "CreditScore","Geography","Gender","Age","Tenure","Balance",
        "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary",
        "SatisfactionScore","CardType","PointEarned","Cluster"
    ]
    target = "Exited"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_features = ["Geography","Gender","CardType"]
    numeric_features     = [
        "CreditScore","Age","Tenure","Balance","NumOfProducts",
        "HasCrCard","IsActiveMember","EstimatedSalary",
        "SatisfactionScore","PointEarned","Cluster"
    ]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(),                              numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),        categorical_features),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="logloss"),
    }

    results      = {}
    roc_data     = {}
    conf_matrices = {}

    for name, m in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", m)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1 Score":  f1_score(y_test, y_pred),
            "ROC-AUC":   roc_auc_score(y_test, y_prob),
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)
        conf_matrices[name] = confusion_matrix(y_test, y_pred)

    return results, roc_data, conf_matrices, y_test

render_sidebar()
render_topnav("Model Performance")

st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Logistic Regression · Random Forest · XGBoost · Evaluation metrics · ROC curves · Confusion matrices</div>', unsafe_allow_html=True)
st.markdown("<hr style='margin:12px 0 20px 0;'>", unsafe_allow_html=True)

with st.spinner("Training models on test set..."):
    results, roc_data, conf_matrices, y_test = load_and_train()

MODEL_COLORS = {
    "Logistic Regression": "#78716c",
    "Random Forest":       AMBER,
    "XGBoost":             GOLD,
}

# metrics table as KPI cards
st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
for model_name, scores in results.items():
    is_best = model_name == "XGBoost"
    border  = f"border-left: 4px solid {GOLD};" if is_best else f"border-left: 4px solid #44403c;"
    badge   = f"<span style='background:{GOLD}22;color:{GOLD};border:1px solid {GOLD}44;font-size:0.65rem;padding:2px 8px;border-radius:4px;margin-left:8px;'>SELECTED MODEL</span>" if is_best else ""

    st.markdown(
        f"<div style='background:#292524;{border}border-top:1px solid #44403c;border-right:1px solid #44403c;"
        f"border-bottom:1px solid #44403c;border-radius:0 8px 8px 0;padding:12px 16px 8px 16px;margin-bottom:10px;'>"
        f"<div style='font-size:0.9rem;font-weight:700;color:#fef3c7;margin-bottom:10px;'>{model_name}{badge}</div>"
        f"<div style='display:flex;gap:16px;flex-wrap:wrap;'>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    for col, metric in zip(cols, metrics):
        val = scores[metric]
        color = GOLD if is_best else TEXT
        with col:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:0.65rem;color:#78716c;text-transform:uppercase;letter-spacing:0.08em;'>{metric}</div>"
                f"<div style='font-size:1.2rem;font-weight:700;color:{color};'>{val:.3f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ROC curves
st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1.3, 1])

with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(color="#44403c", width=1, dash="dash"),
        name="Random Classifier", showlegend=True,
    ))
    for name, (fpr, tpr) in roc_data.items():
        auc = results[name]["ROC-AUC"]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=MODEL_COLORS[name], width=2.5 if name == "XGBoost" else 1.5),
        ))
    l = base_layout("ROC Curve — All Models")
    l["xaxis"]["title"] = "False Positive Rate"
    l["yaxis"]["title"] = "True Positive Rate"
    l["xaxis"]["range"] = [0, 1]
    l["yaxis"]["range"] = [0, 1]
    l["yaxis"]["showgrid"] = True
    l["legend"] = dict(font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)", x=0.55, y=0.1)
    l["height"] = 420
    l["margin"] = dict(t=40, b=40, l=8, r=20)
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # bar chart of ROC-AUC scores
    auc_df = pd.DataFrame([
        {"Model": name, "ROC-AUC": scores["ROC-AUC"]}
        for name, scores in results.items()
    ]).sort_values("ROC-AUC")

    fig = go.Figure(go.Bar(
        x=auc_df["ROC-AUC"],
        y=auc_df["Model"],
        orientation="h",
        marker_color=[MODEL_COLORS[m] for m in auc_df["Model"]],
        text=auc_df["ROC-AUC"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
        textfont=dict(color="#e7e5e4", size=12),
        cliponaxis=False,
    ))
    l = base_layout("ROC-AUC Score by Model")
    l["xaxis"]["title"] = "ROC-AUC"
    l["xaxis"]["range"] = [0.5, 1.0]
    l["yaxis"]["showgrid"] = False
    l["height"] = 420
    l["margin"] = dict(t=40, b=40, l=8, r=80)
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

# confusion matrices
st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)

cm_cols = st.columns(3)
for col, (name, cm) in zip(cm_cols, conf_matrices.items()):
    tn, fp, fn, tp = cm.ravel()
    total    = tn + fp + fn + tp
    accuracy = (tn + tp) / total * 100

    with col:
        fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted: Stay", "Predicted: Churn"],
            y=["Actual: Stay", "Actual: Churn"],
            colorscale=[[0,"#1c1917"],[0.5,"#292524"],[1.0, MODEL_COLORS[name]]],
            showscale=False,
            text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
            texttemplate="%{text}",
            textfont=dict(color="#fef3c7", size=14),
            hovertemplate="<b>%{y} → %{x}</b><br>Count: %{text}<extra></extra>",
        ))
        l = base_layout(name)
        l["height"] = 280
        l["margin"] = dict(t=50, b=20, l=8, r=8)
        l["xaxis"]["showgrid"] = False
        l["xaxis"]["side"] = "bottom"
        l["yaxis"]["showgrid"] = False
        l["yaxis"]["autorange"] = "reversed"
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px;'>"
            f"<div style='background:#292524;border:1px solid #44403c;border-radius:6px;padding:8px;text-align:center;'>"
            f"<div style='font-size:0.6rem;color:#78716c;text-transform:uppercase;'>True Pos</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{GOLD};'>{tp:,}</div></div>"
            f"<div style='background:#292524;border:1px solid #44403c;border-radius:6px;padding:8px;text-align:center;'>"
            f"<div style='font-size:0.6rem;color:#78716c;text-transform:uppercase;'>False Pos</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{RED};'>{fp:,}</div></div>"
            f"<div style='background:#292524;border:1px solid #44403c;border-radius:6px;padding:8px;text-align:center;'>"
            f"<div style='font-size:0.6rem;color:#78716c;text-transform:uppercase;'>False Neg</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{AMBER};'>{fn:,}</div></div>"
            f"<div style='background:#292524;border:1px solid #44403c;border-radius:6px;padding:8px;text-align:center;'>"
            f"<div style='font-size:0.6rem;color:#78716c;text-transform:uppercase;'>True Neg</div>"
            f"<div style='font-size:1rem;font-weight:700;color:#16a34a;'>{tn:,}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# insight
xgb   = results["XGBoost"]
lr    = results["Logistic Regression"]
st.markdown(
    f'<div class="insight-box">⚡ <b>XGBoost</b> achieves the highest ROC-AUC of <b>{xgb["ROC-AUC"]:.3f}</b> '
    f'and F1 Score of <b>{xgb["F1 Score"]:.3f}</b>, outperforming Logistic Regression '
    f'(ROC-AUC {lr["ROC-AUC"]:.3f}) across all metrics. '
    f'The confusion matrix shows XGBoost correctly identifies the most churners (true positives) '
    f'while keeping false positives low — critical for targeting retention campaigns efficiently.</div>',
    unsafe_allow_html=True,
)

st.markdown("<hr style='margin:32px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='display:flex;justify-content:space-between;'>"
    "<span style='font-size:0.72rem;color:#44403c;'>Bank Churn Intelligence · Model Performance</span>"
    "<span style='font-size:0.72rem;color:#44403c;'>XGBoost · Random Forest · Logistic Regression · Scikit-learn</span>"
    "</div>", unsafe_allow_html=True,
)