import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load CSV
df = pd.read_csv("churn_model_comparison.csv")

st.title("Customer Churn Prediction Dashboard")
st.write("Compare Logistic Regression, SMOTE Logistic, and XGBoost predictions")

model_choice = st.sidebar.selectbox(
    "Select Model View",
    ("Logistic", "SMOTE Logistic", "XGBoost")
)

pred_col = f"{model_choice}_Pred"
proba_col = f"{model_choice}_Prob"

top_churners = df[df[pred_col] == 1].sort_values(by=proba_col, ascending=False)

# Metrics
y_true = df['Actual']
y_pred = df[pred_col]
y_prob = df[proba_col]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)

st.subheader(f"Model Metrics: {model_choice}")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**ROC-AUC:** {roc_auc:.2f}")

# Table of top churners
st.subheader(f"Top Churners Predicted by {model_choice}")
st.dataframe(top_churners[['customerID', 'gender', 'tenure', 'MonthlyCharges', 'TotalCharges', pred_col, proba_col]])

# Bar chart: Monthly Charges of top churners
fig1 = px.bar(top_churners.head(20), x='customerID', y='MonthlyCharges', color='MonthlyCharges',
              title=f"Monthly Charges of Top 20 Predicted Churners ({model_choice})")
st.plotly_chart(fig1, use_container_width=True)

# Bar chart: Total Charges of top churners
fig2 = px.bar(top_churners.head(20), x='customerID', y='TotalCharges', color='TotalCharges',
              title=f"Total Charges of Top 20 Predicted Churners ({model_choice})")
st.plotly_chart(fig2, use_container_width=True)
