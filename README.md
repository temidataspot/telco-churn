# Telco Customer Churn Dashboard

A comprehensive customer churn prediction and analysis dashboard for a Telco company. This project implements multiple machine learning models to predict customer churn, compares their performance, and provides an interactive dashboard for exploring high-risk customers. The system helps businesses retain customers, optimise marketing strategies, and prioritise interventions.

Access Dashboard

[![WebApp](https://img.shields.io/badge/Web%20App-Streamlit-pink)](https://churndata.streamlit.app/)

---

## Project Overview

When a customer stops using a company’s service, it is a major challenge 
This project leverages machine learning and interactive data visualisation to:

- **Predict churn using multiple models:**
  - Logistic Regression
  - SMOTE-enhanced Logistic Regression
  - XGBoost Classifier

- **Compare model performance via key metrics:**
  - Accuracy
  - Recall (Churn)
  - ROC-AUC

- **Identify and visualise top churners** based on predicted probability and business features.

- **Provide an interactive dashboard** for filtering, sorting, and exploring customers by features like:
  - Internet Service Type
  - Payment Method
  - Phone Service

---

## 🛠 Tools & Technologies

| Tool | Purpose | 
|------|---------|
| Python | Core programming language | 
| Google Colab | Interactive Coding | 
| NumPy | Numerical computing | 
| Scikit-learn | Machine learning modelling, metrics, preprocessing |
| Altair | Interactive data visualisation in charts |
| Streamlit | Web-based dashboard interface | 
| Git & GitHub | Version control and deployment |

---
## **Machine Learning Models:**

| Model | Details |
|-------|---------|
| Logistic Regression | Baseline model for interpretability |
| SMOTE Logistic Regression | Handles class imbalance using SMOTE oversampling |
| XGBoost Classifier | Gradient boosting for improved predictive performance |


## Notebook & Files

[Full Modelling Notebook](https://github.com/temidataspot/telco-churn/blob/main/Telco_Churn.ipynb)

[Raw Dataset](https://github.com/temidataspot/telco-churn/blob/main/WA_Fn-UseC_-Telco-Customer-Churn.csv)

[Combined Model Predictions CSV](https://github.com/temidataspot/telco-churn/blob/main/churn_model_comparison.csv)

[Dashboard App Code](https://github.com/temidataspot/telco-churn/blob/main/app.py)




## Browse by Categories

<div style="display: flex; gap: 10px; flex-wrap: wrap;">

<a href="https://example.com/health" style="text-decoration: none;">
  <div style="background-color: #d1f7d6; padding: 20px; border-radius: 10px; text-align: center; width: 120px; color: black; font-weight: bold;">
    Health
  </div>
</a>

<a href="https://example.com/finance" style="text-decoration: none;">
  <div style="background-color: #f7f1d1; padding: 20px; border-radius: 10px; text-align: center; width: 120px; color: black; font-weight: bold;">
    Finance
  </div>
</a>

<a href="https://example.com/media" style="text-decoration: none;">
  <div style="background-color: #d1e7f7; padding: 20px; border-radius: 10px; text-align: center; width: 120px; color: black; font-weight: bold;">
    Media
  </div>
</a>

<a href="https://example.com/agric" style="text-decoration: none;">
  <div style="background-color: #f7d1d1; padding: 20px; border-radius: 10px; text-align: center; width: 120px; color: black; font-weight: bold;">
    Agric
  </div>
</a>

</div>

