import streamlit as st

PAGES = [
    ("Dashboard.py",                 "🏠", "Overview"),
    ("pages/1_Churn_Analysis.py",    "📊", "Churn Analysis"),
    ("pages/2_Customer_Segments.py", "👥", "Customer Segments"),
    ("pages/3_Prediction_Centre.py", "🔮", "Prediction Centre"),
    ("pages/4_Retention_Centre.py",  "🎯", "Retention Centre"),
]

COMMON_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #1c1917; color: #e7e5e4; }
  [data-testid="stSidebar"] { background-color: #161412 !important; border-right: 1px solid #292524; }
  [data-testid="stSidebar"] * { color: #a8a29e !important; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  .page-title { font-size: 1.6rem; font-weight: 700; color: #fef3c7; letter-spacing: -0.02em; }
  .page-subtitle { font-size: 0.82rem; color: #57534e; margin-top: 4px; }
  .section-header { font-size: 0.7rem; font-weight: 600; color: #78716c; text-transform: uppercase; letter-spacing: 0.15em; border-bottom: 1px solid #292524; padding-bottom: 8px; margin: 32px 0 16px 0; }
  .kpi-card { background: #292524; border: 1px solid #44403c; border-top: 3px solid #f59e0b; border-radius: 10px; padding: 18px 20px 14px 20px; }
  .kpi-label { font-size: 0.7rem; font-weight: 500; color: #78716c; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
  .kpi-value { font-size: 1.9rem; font-weight: 700; color: #fbbf24; line-height: 1.1; }
  .kpi-sub { font-size: 0.72rem; color: #57534e; margin-top: 5px; }
  .kpi-badge { display: inline-block; font-size: 0.68rem; padding: 2px 8px; border-radius: 4px; margin-top: 5px; background: #1c1917; border: 1px solid #44403c; color: #a8a29e; }
  .insight-box { background: #292524; border: 1px solid #44403c; border-left: 3px solid #f59e0b; border-radius: 8px; padding: 12px 16px; font-size: 0.8rem; color: #a8a29e; line-height: 1.6; }
  .insight-box b { color: #fbbf24; }
  .persona-card { background: #292524; border: 1px solid #44403c; border-top: 3px solid; border-radius: 10px; padding: 18px 20px; }
  .persona-name { font-size: 0.95rem; font-weight: 700; color: #fef3c7; margin-bottom: 4px; }
  .persona-tag { display: inline-block; font-size: 0.65rem; padding: 2px 8px; border-radius: 4px; margin-bottom: 10px; font-weight: 500; }
  .persona-stat-label { font-size: 0.65rem; color: #78716c; text-transform: uppercase; letter-spacing: 0.08em; }
  .persona-stat-value { font-size: 1.1rem; font-weight: 700; color: #fbbf24; }
  .persona-desc { font-size: 0.75rem; color: #a8a29e; line-height: 1.6; margin-top: 10px; border-top: 1px solid #44403c; padding-top: 10px; }
  .topnav { display: flex; align-items: center; gap: 4px; background: #161412; border: 1px solid #292524; border-radius: 10px; padding: 8px 14px; margin-bottom: 20px; flex-wrap: wrap; }
  .topnav-sep { color: #44403c; font-size: 0.75rem; padding: 0 2px; }
  hr { border-color: #292524 !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .stSelectbox > div > div { background-color: #292524 !important; border-color: #44403c !important; color: #e7e5e4 !important; }
</style>
"""

def render_sidebar(extra_filters=None):
    with st.sidebar:
        st.markdown("<div style='padding:8px 0 16px 0;'><span style='font-size:1.1rem;font-weight:700;color:#fbbf24 !important;'>🏦 Bank Churn Intelligence</span></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:0 0 16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#57534e;margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
        for path, icon, label in PAGES:
            st.page_link(path, label=f"◆  {label}")
        if extra_filters:
            st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#57534e;margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)
            extra_filters()
        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.68rem;color:#44403c;'>Portfolio Project · 2026</div>", unsafe_allow_html=True)


def render_topnav(current: str):
    cols = st.columns(len(PAGES))
    for col, (path, icon, label) in zip(cols, PAGES):
        with col:
            if label == current:
                st.markdown(
                    f"<div style='text-align:center;padding:6px 4px;background:#292524;"
                    f"border:1px solid #44403c;border-bottom:2px solid #f59e0b;"
                    f"border-radius:8px;font-size:0.78rem;font-weight:600;color:#fbbf24;'>"
                    f"{icon} {label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.page_link(path, label=f"{icon}  {label}")


def base_layout(title="", r_margin=80):
    STONE = "#44403c"; TEXT = "#a8a29e"
    return dict(
        title=dict(text=title, font=dict(color="#e7e5e4", size=13), x=0, xanchor="left", pad=dict(b=8)),
        paper_bgcolor="#1c1917", plot_bgcolor="#1c1917",
        font=dict(color=TEXT, family="Inter", size=11),
        margin=dict(t=40, b=32, l=8, r=r_margin),
        xaxis=dict(showgrid=False, color=TEXT, linecolor=STONE, tickcolor=STONE),
        yaxis=dict(showgrid=True, gridcolor="#292524", color=TEXT, linecolor=STONE),
    )

GOLD  = "#f59e0b"
GOLD2 = "#fbbf24"
RED   = "#dc2626"
AMBER = "#d97706"
STONE = "#44403c"
TEXT  = "#a8a29e"

CLUSTER_LABELS = {0:"Active High-Balance", 1:"Older At-Risk", 2:"Inactive High-Balance", 3:"Low-Balance Product-Heavy"}
CLUSTER_COLORS = {0:"#f59e0b", 1:"#dc2626", 2:"#d97706", 3:"#78716c"}
CLUSTER_RISK   = {0:"Low Risk", 1:"High Risk", 2:"Medium-High Risk", 3:"Low Risk"}
CLUSTER_RISK_COLORS = {0:"#16a34a", 1:"#dc2626", 2:"#d97706", 3:"#16a34a"}
CLUSTER_DESC = {
    0: "Younger customers with high balances who are actively engaged. Lower churn rate suggests strong product fit. Priority: maintain engagement and cross-sell.",
    1: "Oldest customer group with moderately high balances and the highest churn rate at 36%. These customers may have unmet financial needs or competitive alternatives. Priority: immediate personalised outreach.",
    2: "High-balance customers who are disengaged. Despite financial value, inactivity signals risk. Priority: re-engagement campaigns before they leave.",
    3: "Younger customers with low balances but higher product ownership. Low churn and high engagement. Priority: nurture loyalty and grow wallet share over time.",
}