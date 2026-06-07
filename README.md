# Bank Customer Churn Prediction & Retention Dashboard

## Executive Summary

Customer retention is one of the most important challenges faced by banks. Acquiring a new customer is often more expensive than retaining an existing one. This project aims to identify customers who are likely to churn, understand the key drivers behind churn behaviour, estimate potential business impact, and provide data-driven retention recommendations.

The final solution will combine exploratory data analysis, customer segmentation, machine learning, survival analysis, and interactive business dashboards to support customer retention strategies.

---

## Project Objectives

-  Identify factors that contribute to customer churn.
- Predict which customers are likely to leave the bank.
- Estimate revenue at risk from potential churn.
- Segment customers into meaningful groups and personas.
- Recommend targeted retention actions.
- Deliver insights through an interactive Streamlit dashboard.

---

## Dataset

**Dataset:** Bank Customer Churn Dataset

### Current Dataset Overview

* 10,000 customer records
* 18 features
* No missing values
* No duplicate records
* Churn rate: 20.38%

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
│   ├── 02_data_cleaning_feature_engineering
│   ├── 03_automated_eda.ipynb
│   └── 04_custom_churn_eda.ipynb
│
├── pages/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Planned Features

#### Business Analytics

* Customer churn overview
* Churn driver analysis
* Revenue at risk analysis

#### Customer Intelligence

* Rule-based customer segmentation
* K-Means customer personas

#### Predictive Analytics

* Logistic Regression
* Random Forest
* XGBoost
* Cox Proportional Hazards Survival Model

#### Decision Intelligence

* Churn risk scoring
* Retention recommendation engine

#### Dashboard

* Executive overview
* Customer segmentation
* Churn prediction
* Retention centre
* Customer explorer

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Improvements

* Deploy the dashboard publicly.
* Connect to a live database.
* Add automated reporting.
* Add real-time customer monitoring.
* Integrate customer lifetime value estimation.

---

## Author

Portfolio Project – Customer Churn Prediction & Retention Analytics
