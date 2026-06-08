# Bank Customer Churn Prediction & Retention Dashboard

## Executive Summary

Customer retention is a major challenge for financial institutions, as acquiring new customers is significantly more expensive than retaining existing ones. This project leverages data analytics and machine learning to identify customers at risk of churn, understand the factors driving customer attrition, estimate potential revenue exposure, and provide actionable retention strategies.

The solution combines exploratory data analysis, customer segmentation, predictive modelling, revenue-at-risk analysis, and an interactive business dashboard to support data-driven decision-making.

---

## Project Objectives

- Identify factors that contribute to customer churn
- Predict which customers are likely to leave the bank
- Estimate revenue at risk from potential churn
- Segment customers into meaningful groups and personas
- Provide retention-focused business recommendations
- Deliver insights through an interactive Streamlit dashboard

---

## Dataset

**Dataset:** Bank Customer Churn Dataset

### Current Dataset Overview

- 10,000 customer records
- 18 original features
- 4 engineered features
- 1 customer segmentation feature
- No missing values
- No duplicate records
- Churn rate: 20.38%

### Target Variable

| Variable | Description                                 |
| -------- | ------------------------------------------- |
| Exited   | 1 = Customer Churned, 0 = Customer Retained |

---

## Tech Stack

- Python
- Pandas
- NumPy
- Streamlit
- Scikit-learn
- XGBoost
- Plotly
- Matplotlib
- Seaborn
- Lifelines
- GitHub

---

## Project Structure

```text
bank-churn-intelligence/
│
├── assets/
├── data/
│   └── bank_customer_churn.csv
│
├── models/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_feature_engineering.ipynb
│   ├── 03_custom_churn_eda.ipynb
│   ├── 04_customer_segmentation.ipynb
│   ├── 05_churn_prediction_model.ipynb
│   └── 06_revenue_at_risk.ipynb
│
├── pages/
│   ├── 1_Churn_Analysis.py
│   ├── 2_Customer_Segments.py
│   ├── 3_Prediction_Centre.py
│   └── 4_Retention_Centre.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Implemented Features

### Data Understanding & Preparation
- Dataset profiling
- Data quality assessment
- Feature engineering
- Customer behaviour categorisation

### Exploratory Data Analysis
- Churn distribution analysis
- Geography-based churn analysis
- Demographic churn analysis
- Product and balance behaviour analysis
- Customer satisfaction analysis

### Customer Segmentation
- K-Means clustering
- Customer persona identification
- Cluster profiling

### Predictive Modelling
- Logistic Regression
- Random Forest
- XGBoost
- Churn probability scoring

### Revenue at Risk Analysis
- Customer value estimation
- Revenue at risk calculation
- Risk tier classification
- Revenue exposure analysis by geography and customer segment

## TBD:

### Executive Dashboard
- Overall churn KPIs
- Revenue at risk KPIs
- Churn trend monitoring

### Customer Intelligence
- Customer segmentation explorer
- Cluster-level insights

### Churn Prediction Centre
- Individual customer churn prediction
- Churn probability scoring

### Retention Centre
- High-risk customer identification
- Retention recommendations
- Revenue recovery opportunities

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run Dashboard.py
```

---

## Future Improvements

- Deploy the application to the cloud
- Connect to a live database
- Add automated business reporting
- Implement real-time customer monitoring
- Integrate advanced customer lifetime value modelling using transaction-level history
- Add explainable AI using SHAP to interpret churn predictions

---

## Author

Portfolio Project – Customer Churn Prediction & Retention Analytics
