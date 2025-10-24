# 📊 Telco Customer Churn Dashboard

![Telco Churn](https://churndata.streamlit.app/800x200.png?text=Telco+Customer+Churn+Dashboard)

**Compare predictions from multiple machine learning models and explore top churners interactively.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-orange)](https://docs.streamlit.io/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-green)](https://scikit-learn.org/stable/)  
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)](https://xgboost.readthedocs.io/)  
[![Altair](https://img.shields.io/badge/Altair-5.0-yellow)](https://altair-viz.github.io/)  

---

## 🔗 Overview

The **Telco Customer Churn Dashboard** predicts customer churn, compares multiple models, and visualises high-risk customers using interactive charts and tables.  

**Objectives:**  
- Predict customer churn using **Logistic Regression, SMOTE Logistic, and XGBoost**  
- Evaluate and compare **model metrics**  
- Identify **top churners** based on probability and business metrics  
- Provide **interactive filtering** by Internet Service, Payment Method, and Phone Service  

---

## 📊 Metrics Overview

<div style="display:flex; gap:30px;">
<div style="text-align:center;">
  <h3>Accuracy</h3>
  <p style="font-size:24px;"><b>0.78</b></p>
</div>
<div style="text-align:center;">
  <h3>Recall (Churn)</h3>
  <p style="font-size:24px;"><b>0.64</b></p>
</div>
<div style="text-align:center;">
  <h3>ROC-AUC</h3>
  <p style="font-size:24px;"><b>0.84</b></p>
</div>
</div>

> Metrics dynamically update in the dashboard based on **selected model and filters**.

---

## 📈 Dashboard Visuals

The dashboard includes **4 key visuals** to explore top churners:

| Chart | Description | Example |
|-------|-------------|---------|
| **Total Charges by Top Churners** | High-paying customers at risk | ![TotalCharges](https://via.placeholder.com/300x200.png?text=TotalCharges) |
| **Top Churners by Phone Service** | Distribution of phone services among churners | ![PhoneService](https://via.placeholder.com/300x200.png?text=PhoneService) |
| **Top Churners by Payment Method** | Payment preference among churners | ![PaymentMethod](https://via.placeholder.com/300x200.png?text=PaymentMethod) |
| **Top Churners by Internet Service** | Internet service usage among churners | ![InternetService](https://via.placeholder.com/300x200.png?text=InternetService) |

> Interactive filters allow you to select a **custom number of top churners**, specific **internet services**, **payment methods**, or **phone services**. Selecting **“Select All”** ensures all options are included in analysis.

---

## 🗂 Interactive Tables

The dashboard provides two interactive tables:  
1. **Top Churners Details** – Displays customers predicted to churn with their probability, total charges, and account features.  
2. **Filtered Data Preview** – Shows all customers based on applied filters for deeper exploration.  

![TablePreview](https://via.placeholder.com/600x200.png?text=Data+Preview)

---

## 🛠 Tools & Technologies

- **Python 3.12** – Core programming language  
- **Pandas / NumPy** – Data cleaning & manipulation  
- **Scikit-learn** – Modeling and metrics  
- **Imbalanced-learn (SMOTE)** – Class imbalance handling  
- **XGBoost** – Gradient boosting model  
- **Altair** – Interactive charts  
- **Streamlit** – Web dashboard interface  

---

## 🔍 Key Insights

- **High churn risk** customers often have **Fiber Optic internet** and **high total charges**  
- **SMOTE Logistic** improves recall at minor accuracy cost  
- **XGBoost** provides a good balance between **recall** and **ROC-AUC**, ideal for identifying potential churners  

---

## 🚀 How to Run the Dashboard

1. **Clone the repository**:  
```bash
git clone https://github.com/your_username/telco-churn.git
cd telco-churn
