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

# --- Centered Title & Subtitle ---
st.markdown("<h1 style='text-align: center; color: white;'>Telco Customer Churn Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: gray;'>Compare predictions from different models and explore top churners.</h4>", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("churn_model_comparison.csv")
    return df

df = load_data()

# --- Sidebar Filters ---
# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model View:",
    ["Logistic Regression", "SMOTE Logistic", "XGBoost"]
)

# Utility to add "Select All" at the top
def multiselect_with_all(label, options):
    default = options  # all selected by default
    selected = st.sidebar.multiselect(label, options=options, default=default)
    if not selected:
        selected = options  # fallback to all if user deselects everything
    return selected

# Multi-select filters with select all by default
internet_filter = multiselect_with_all(
    "Filter by Internet Service:",
    df['InternetService'].unique()
)

payment_filter = multiselect_with_all(
    "Filter by Payment Method:",
    df['PaymentMethod'].unique()
)

phone_filter = multiselect_with_all(
    "Filter by Phone Service:",
    df['PhoneService'].unique()
)

# Apply filters
df_filtered = df[
    (df['InternetService'].isin(internet_filter)) &
    (df['PaymentMethod'].isin(payment_filter)) &
    (df['PhoneService'].isin(phone_filter))
]

# --- Map choice to column names ---
model_pred_col = {
    "Logistic Regression": ("Logistic_Pred", "Logistic_Prob"),
    "SMOTE Logistic": ("Smote_Pred", "Smote_Prob"),
    "XGBoost": ("XGB_Pred", "XGB_Prob")
}

pred_col, prob_col = model_pred_col[model_choice]

# --- Metrics ---
accuracy = (df_filtered['Actual'] == df_filtered[pred_col]).mean()
recall = df_filtered[df_filtered[pred_col]==1]['Actual'].sum() / df_filtered['Actual'].sum()
roc_auc = df_filtered[[pred_col,'Actual']].corr().iloc[0,1]

st.markdown("### Model Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Recall (Churn)", f"{recall:.2f}")
col3.metric("ROC-AUC (Approx.)", f"{roc_auc:.2f}")

# --- Top Churners ---
top_n = st.sidebar.slider("Select number of top churners to view", 5, 50, 10)
top_churners = df_filtered.sort_values(prob_col, ascending=False).head(top_n)

# --- Visuals ---
st.markdown("### Visual Analysis")
vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.markdown("**Total Charges of Top Churners**")
    sns.barplot(x='customerID', y='TotalCharges', data=top_churners)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

with vis_col2:
    st.markdown("**Top Churners with Phone Service**")
    phone_counts = top_churners['PhoneService'].value_counts()
    plt.pie(phone_counts, labels=phone_counts.index, autopct='%1.1f%%', colors=['#4B0082','#9370DB'])
    st.pyplot(plt.gcf())
    plt.clf()

vis_col3, vis_col4 = st.columns(2)

with vis_col3:
    st.markdown("**Top Churners by Payment Method**")
    sns.countplot(y='PaymentMethod', data=top_churners, palette='cool')
    st.pyplot(plt.gcf())
    plt.clf()

with vis_col4:
    st.markdown("**Top Churners by Internet Service**")
    sns.countplot(y='InternetService', data=top_churners, palette='magma')
    st.pyplot(plt.gcf())
    plt.clf()

# --- Table ---
st.markdown("### Top Churners Table")
st.dataframe(top_churners[['customerID', 'gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Actual', pred_col, prob_col, 'InternetService', 'PaymentMethod', 'PhoneService']])
