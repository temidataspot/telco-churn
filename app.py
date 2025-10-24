import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import recall_score, roc_auc_score

# --- Load Data ---
df = pd.read_csv("churn_model_comparison.csv")

# --- Page Setup ---
st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide")

# --- Title and Subtitle ---
st.markdown("<h1 style='text-align: center;'>Telco Customer Churn Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Compare predictions from different models and explore top churners</h4>", unsafe_allow_html=True)
st.write("---")

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model View:",
    ["Logistic Regression", "SMOTE Logistic", "XGBoost"]
)

# Top N slider
top_n = st.sidebar.slider("Top N Churners", min_value=5, max_value=50, value=20)

# Multi-select with explicit "Select All"
def multiselect_with_select_all(label, options):
    options_with_all = ["Select All"] + list(options)
    selected = st.sidebar.multiselect(label, options_with_all, default=["Select All"])
    if "Select All" in selected:
        return list(options) # all values
    else:
        return selected

internet_filter = multiselect_with_select_all("Filter by Internet Service:", df['InternetService'].unique())
payment_filter = multiselect_with_select_all("Filter by Payment Method:", df['PaymentMethod'].unique())
phone_filter = multiselect_with_select_all("Filter by Phone Service:", df['PhoneService'].unique())

# --- Map selected model to prediction columns ---
if model_choice == "Logistic Regression":
    pred_col = "Logistic_Pred"
    prob_col = "Logistic_Prob"
elif model_choice == "SMOTE Logistic":
    pred_col = "Smote_Pred"
    prob_col = "Smote_Prob"
elif model_choice == "XGBoost":
    pred_col = "XGB_Pred"
    prob_col = "XGB_Prob"
else:
    st.error("Invalid model selected")
    st.stop()

# --- Apply filters ---
df_filtered = df[
    df['InternetService'].isin(internet_filter) &
    df['PaymentMethod'].isin(payment_filter) &
    df['PhoneService'].isin(phone_filter)
]

if df_filtered.empty:
    st.warning("No data matches the current filter selection!")
    st.stop()

# --- Metrics ---
accuracy = (df_filtered['Actual'] == df_filtered[pred_col]).mean()
recall = recall_score(df_filtered['Actual'], df_filtered[pred_col])
roc_auc = roc_auc_score(df_filtered['Actual'], df_filtered[prob_col])

# Display metrics in 3 columns
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Recall (Churn)", f"{recall:.2f}")
col3.metric("ROC-AUC", f"{roc_auc:.2f}")

st.write("---")

# --- Top Churners ---
top_churners = df_filtered[df_filtered[pred_col] == 1].sort_values(prob_col, ascending=False).head(top_n)

# --- Charts ---
# Total charges by top churners (bar)
chart1 = alt.Chart(top_churners).mark_bar(color="#FF6F61").encode(
    x=alt.X("customerID:N", title="Customer ID"),
    y=alt.Y("TotalCharges:Q", title="Total Charges"),
    tooltip=["customerID", "TotalCharges"]
).properties(title="Total Charges by Top Churners")

# Percentage of top churners with phone service (pie)
phone_counts = top_churners['PhoneService'].value_counts(normalize=True).reset_index()
phone_counts.columns = ['PhoneService', 'Percentage']
chart2 = alt.Chart(phone_counts).mark_arc(innerRadius=50).encode(
    theta=alt.Theta("Percentage:Q"),
    color=alt.Color("PhoneService:N"),
    tooltip=["PhoneService", alt.Tooltip("Percentage:Q", format=".0%")]
).properties(title="Top Churners by Phone Service")

# Top churners by payment method (bar)
payment_counts = top_churners['PaymentMethod'].value_counts().reset_index()
payment_counts.columns = ['PaymentMethod', 'Count']
chart3 = alt.Chart(payment_counts).mark_bar(color="#6A5ACD").encode(
    x=alt.X("PaymentMethod:N", title="Payment Method"),
    y=alt.Y("Count:Q", title="Number of Top Churners"),
    tooltip=["PaymentMethod", "Count"]
).properties(title="Top Churners by Payment Method")

# Top churners by internet service (bar)
internet_counts = top_churners['InternetService'].value_counts().reset_index()
internet_counts.columns = ['InternetService', 'Count']
chart4 = alt.Chart(internet_counts).mark_bar(color="#20B2AA").encode(
    x=alt.X("InternetService:N", title="Internet Service"),
    y=alt.Y("Count:Q", title="Number of Top Churners"),
    tooltip=["InternetService", "Count"]
).properties(title="Top Churners by Internet Service")

# Display charts 2x2
col1, col2 = st.columns(2)
col1.altair_chart(chart1, use_container_width=True)
col2.altair_chart(chart2, use_container_width=True)

col3, col4 = st.columns(2)
col3.altair_chart(chart3, use_container_width=True)
col4.altair_chart(chart4, use_container_width=True)

st.write("---")

# --- Tables ---
st.subheader("Top Churners Details")
st.dataframe(top_churners)

st.subheader("Filtered Data Preview")
st.dataframe(df_filtered)
