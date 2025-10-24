# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(
    page_title="Telco Churn Dashboard",
    layout="wide"
)

st.title("Telco Customer Churn Dashboard")
st.markdown("Compare predictions from different models and explore top churners.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("churn_model_comparison.csv")
    return df

df = load_data()

# --- Sidebar: Model Selector ---
model_choice = st.sidebar.selectbox(
    "Select Model View:",
    ["Logistic Regression", "SMOTE Logistic", "XGBoost"]
)

# Map choice to column names
model_pred_col = {
    "Logistic Regression": ("Logistic_Pred", "Logistic_Prob"),
    "SMOTE Logistic": ("Smote_Pred", "Smote_Prob"),
    "XGBoost": ("XGB_Pred", "XGB_Prob")
}

pred_col, prob_col = model_pred_col[model_choice]

# --- Metrics ---
accuracy = df[df[pred_col].notna()].apply(
    lambda row: 1 if row['Actual'] == row[pred_col] else 0, axis=1
).mean()

recall = df[df[pred_col]==1]['Actual'].sum() / df['Actual'].sum()
roc_auc = df[[pred_col, 'Actual']].corr().iloc[0,1]  # rough proxy

st.subheader(f"Model Metrics for {model_choice}")
st.metric("Accuracy", f"{accuracy:.2f}")
st.metric("Recall (Churn)", f"{recall:.2f}")
st.metric("ROC-AUC (Approx.)", f"{roc_auc:.2f}")

# --- Filter top churners ---
top_n = st.sidebar.slider("Select number of top churners to view", 5, 50, 10)
top_churners = df.sort_values(prob_col, ascending=False).head(top_n)

st.subheader(f"Top {top_n} Predicted Churners ({model_choice})")
st.dataframe(top_churners[['customerID', 'gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Actual', pred_col, prob_col]])

# --- Visuals ---
st.subheader("Visual Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Churn Probability Distribution")
    sns.histplot(df[prob_col], bins=20, kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

with col2:
    st.markdown("### Total Charges of Top Churners")
    sns.barplot(x='customerID', y='TotalCharges', data=top_churners)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

st.markdown("---")
st.write("Use the sidebar to select different model views and number of top churners.")
