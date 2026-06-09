# 🏦 Bank Churn Intelligence

> End-to-end customer churn prediction and retention analytics dashboard built with Python, XGBoost, SHAP, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC%200.875-gold) ![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)

🔗 **[Live Demo](https://bank-churn-intel.streamlit.app/)**

---

## Executive Summary

Customer retention is a major challenge for financial institutions as acquiring new customers costs significantly more than retaining existing ones. This project applies data analytics and machine learning to:

- Identify which customers are likely to churn
- Understand the behavioural and demographic drivers of attrition
- Estimate revenue exposure from predicted churn
- Segment customers into actionable personas
- Deliver explainable predictions and retention recommendations through an interactive business dashboard

---

## Screenshots

### Overview Dashboard
![Overview](screenshots/01_overview_kpi.png)
*Headline KPIs, churn overview charts, and revenue at risk by geography and customer persona*

### Churn Analysis
![Churn Analysis](screenshots/04_churn_analysis_heatmap.png)
*Cross-dimensional churn rate analysis revealing high-risk age and balance combinations*

### Customer Segments
![Segments](screenshots/06_segments_radar.png)
*Normalised radar chart comparing 4 K-Means customer personas across 6 behavioural dimensions*

### Prediction Centre – Live SHAP Explainability
![Prediction](screenshots/08_prediction_shap.png)
*Real-time SHAP waterfall chart showing individual feature contributions to each churn prediction*

### Retention Centre
![Retention](screenshots/10_retention_table.png)
*Filterable high-risk customer list ranked by revenue at risk with CSV export*

### Model Performance
![Model Performance](screenshots/11_model_metrics_roc.png)
*Side-by-side evaluation of Logistic Regression, Random Forest, and XGBoost with ROC curves*

---

## Project Structure

```text
bank-churn-intelligence/
│
├── data/
│   ├── bank_customer_churn.csv               # Raw dataset (10,000 records)
│   ├── cleaned_bank_churn.csv                # After renaming, dropping cols, feature engineering
│   ├── segmented_bank_churn.csv              # After K-Means clustering (adds Cluster column)
│   ├── churn_predictions.csv                 # Model output: churn probabilities & risk categories
│   └── revenue_at_risk.csv                   # Revenue at risk calculations per customer
│
├── models/
│   ├── churn_prediction_model.joblib         # Saved XGBoost pipeline (preprocessor + model)
│   ├── shap_explainer.pkl                    # SHAP TreeExplainer for individual predictions
│   ├── shap_feature_names.pkl                # Feature names post-preprocessing (20 features)
│   └── model_performance_results.pkl         # Pre-computed metrics, ROC data, confusion matrices
│
├── notebooks/
│   ├── 01_data_understanding.ipynb           # Dataset profiling and quality assessment
│   ├── 02_data_cleaning_feature_engineering.ipynb  # Cleaning and feature creation
│   ├── 03_custom_churn_eda.ipynb             # Exploratory data analysis
│   ├── 04_customer_segmentation.ipynb        # K-Means clustering and persona identification
│   ├── 05_churn_prediction_model.ipynb       # Model training, evaluation, SHAP export
│   └── 06_revenue_at_risk.ipynb              # Revenue at risk framework
│
├── pages/
│   ├── 1_Churn_Analysis.py                   # Demographic, behavioural and product churn patterns
│   ├── 2_Customer_Segments.py                # Customer personas and segment explorer
│   ├── 3_Prediction_Centre.py                # Live churn prediction with SHAP explainability
│   ├── 4_Retention_Centre.py                 # At-risk customers and retention playbook
│   └── 5_Model_Performance.py                # ROC curves, confusion matrices, metric comparison
│
├── screenshots/
│   ├── dashboard_overview.png
│   ├── dashboard_filters.png
│   ├── churn_analysis_demographics.png
│   ├── churn_analysis_heatmap.png
│   ├── customer_segments_personas.png
│   ├── customer_segments_radar.png
│   ├── prediction_low_risk.png
│   ├── prediction_high_risk.png
│   ├── prediction_shap.png
│   ├── retention_playbook.png
│   ├── retention_table.png
│   ├── model_performance_metrics.png
│   ├── model_performance_roc.png
│   └── model_performance_confusion.png
│
├── app.py                              # Main entry point and overview page
├── utils.py                                  # Shared CSS, navigation, chart helpers
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Source:** Bank Customer Churn Dataset (Kaggle)

| Property | Value |
|---|---|
| Records | 10,000 customers |
| Original features | 18 |
| Engineered features | 4 (AgeGroup, BalanceGroup, TenureGroup, SalaryGroup) |
| Segmentation feature | 1 (Cluster) |
| Churn rate | 20.38% |
| Missing values | None |
| Duplicates | None |

**Target variable:** `Exited` – 1 = Churned, 0 = Retained

---

## Dashboard Pages

### 🏠 Overview
Headline KPIs including total customers, historical churn rate, predicted churners, high-risk count, and total revenue at risk. Churn overview charts, risk distribution, and key churn driver breakdowns by age, geography, and product usage. Includes geography and risk tier filters.

### 📊 Churn Analysis
Deep-dive into churn patterns across demographics (gender, age, geography), behaviour (activity status, complaints, satisfaction), and product usage (number of products, balance group, card type). Includes an age × balance churn heatmap as the showstopper visual.

### 👥 Customer Segments
K-Means clustering with k=4, producing four customer personas:

| Cluster | Persona | Churn Risk |
|---|---|---|
| 0 | Active High-Balance | Low (13%) |
| 1 | Older At-Risk | High (36%) |
| 2 | Inactive High-Balance | Medium-High (29%) |
| 3 | Low-Balance Product-Heavy | Low (12%) |

Includes persona cards with risk badges, normalised radar chart, segment comparison charts, and stacked churn breakdown.

### 🔮 Prediction Centre
Live individual churn prediction using the saved XGBoost pipeline. Input any customer profile and receive:
- Real-time churn probability with gauge chart
- Risk category badge (Low / Medium / High)
- **SHAP waterfall chart** showing actual model feature contributions
- Similar customer comparison with distribution chart and insight

### 🎯 Retention Centre
Business-focused retention toolkit:
- Filtered at-risk customer table (adjustable probability threshold and segment)
- Revenue exposure charts by region and segment
- Churn probability vs balance scatter plot
- Retention playbook with prioritised strategies per segment
- Downloadable at-risk customer CSV

### 📈 Model Performance
Full model evaluation across three candidate models (loads instantly from pre-computed results):
- Metric comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)
- ROC curves for all three models on one chart
- Confusion matrices with TP/FP/FN/TN breakdown

---

## Modelling

### Models Evaluated

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.817 | 0.661 | 0.206 | 0.314 | 0.778 |
| Random Forest | 0.851 | 0.642 | 0.610 | 0.626 | 0.865 |
| **XGBoost** ✓ | **0.868** | **0.775** | **0.498** | **0.606** | **0.875** |

**XGBoost** was selected as the final model based on highest ROC-AUC (0.875) and best overall predictive performance.

### Features Used

```
CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
HasCrCard, IsActiveMember, EstimatedSalary, SatisfactionScore,
CardType, PointEarned, Cluster
```

> **Note on data leakage:** The `Complain` feature was excluded from modelling. Initial experiments produced near-perfect metrics (ROC-AUC ~1.0), indicating it was a near-direct proxy for the target variable rather than a genuine behavioural predictor.

### Explainability
SHAP (SHapley Additive exPlanations) via `TreeExplainer` explains individual predictions. The dashboard renders a real-time SHAP waterfall chart for any input profile, showing which features push the prediction toward or away from churn. Model baseline (average churn probability): ~20.1%.

---

## Revenue at Risk Framework

Since the dataset lacks transaction-level history, a simplified customer value proxy is used:

```
CustomerValueScore = 0.40 × BalanceScore
                   + 0.30 × SalaryScore
                   + 0.20 × TenureScore
                   + 0.10 × PointsScore

EstimatedCustomerValue = CustomerValueScore × EstimatedSalary
RevenueAtRisk          = EstimatedCustomerValue × ChurnProbability
```

**Total revenue at risk across the dataset: ~$105.9M**

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Dashboard | Streamlit, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Data | Pandas, NumPy |
| Clustering | Scikit-learn KMeans |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Version Control | Git, GitHub |

---

## Key Findings

- **Germany** has the highest churn rate at 32.4% vs ~16% for France and Spain
- **Inactive customers** churn at 26.9% vs 14.3% for active members
- **Customers who complained** churn at 99.5% – complaint resolution is the single highest-leverage retention action
- **2-product customers** have the lowest churn at 7.6%; 3+ product customers spike to 82.7%+
- **51–60 age group** has the highest churn at 56.2%
- **Cluster 1 (Older At-Risk)** carries the highest individual churn risk at 36%
- **Cluster 2 (Inactive High-Balance)** carries the highest total revenue at risk at ~$49.8M

---

## How to Run

```bash
git clone https://github.com/Nihira11/bank-churn-intelligence.git
cd bank-churn-intelligence
pip install -r requirements.txt
streamlit run Dashboard.py
```

---

## Future Improvements

- Connect to a live database for real-time monitoring
- Replace CLV proxy with a full discounted cash flow model given richer transaction data
- Add automated business reporting (PDF export)
- Implement customer lifetime value modelling with transaction-level history

---

## Author

Portfolio Project – Customer Churn Prediction & Retention Analytics  
Built as part of a data science portfolio targeting fintech and quantitative analytics roles