# =============================================
# STREAMLIT DASHBOARD
# Credit Risk & Fraud Detection Engine
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Banking ML Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk & Fraud Detection Engine")
st.markdown("*ML-powered banking intelligence with SHAP explainability*")

# --- Tabs ---
tab1, tab2 = st.tabs(["💳 Credit Risk", "🚨 Fraud Detection"])

# =============================================
# TAB 1 - CREDIT RISK
# =============================================
with tab1:
    st.header("Credit Risk Predictor")
    st.markdown("Predict whether a loan applicant is high or low risk")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 75, 35)
        credit_amount = st.slider("Loan Amount ($)", 500, 20000, 5000)
        duration = st.slider("Loan Duration (months)", 6, 72, 24)

    with col2:
        job = st.selectbox("Job Type", [0, 1, 2, 3],
            format_func=lambda x: ["Unskilled", "Unskilled Resident",
                                    "Skilled", "Highly Skilled"][x])
        housing = st.selectbox("Housing", ["own", "free", "rent"])
        saving_accounts = st.selectbox("Saving Accounts",
            ["little", "moderate", "quite rich", "rich"])

    if st.button("Predict Credit Risk", type="primary"):
        # Load and retrain for demo
        try:
            credit_df = pd.read_csv('german_credit_data.csv', index_col=0)
            credit_df.dropna(inplace=True)
            le = LabelEncoder()
            cat_cols = credit_df.select_dtypes(
                include='object').columns.tolist()
            cat_cols = [c for c in cat_cols if c != 'Risk']
            for col in cat_cols:
                credit_df[col] = le.fit_transform(
                    credit_df[col].astype(str))
            credit_df['Risk'] = (
                credit_df['Risk'] == 'bad').astype(int)
            X = credit_df.drop('Risk', axis=1)
            y = credit_df['Risk']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            model = XGBClassifier(n_estimators=100, random_state=42,
                use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)

            # Single prediction
            input_data = X_test[0].reshape(1, -1)
            prob = model.predict_proba(input_data)[0][1]
            prediction = "HIGH RISK 🔴" if prob > 0.5 else "LOW RISK 🟢"

            st.markdown("---")
            st.subheader("Prediction Result")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Level", prediction)
            with col_b:
                st.metric("Risk Probability", f"{prob*100:.1f}%")

            # SHAP
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_data)
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    feature_names=X.columns.tolist()
                ), show=False
            )
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure german_credit_data.csv is in the same folder")

# =============================================
# TAB 2 - FRAUD DETECTION
# =============================================
with tab2:
    st.header("Transaction Fraud Detector")
    st.markdown("Flag suspicious transactions in real time")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.slider("Transaction Amount ($)", 0.0, 5000.0, 150.0)
        hour = st.slider("Hour of Day", 0, 23, 14)
    with col2:
        v1 = st.slider("Risk Score V1", -5.0, 5.0, 0.0)
        v2 = st.slider("Risk Score V2", -5.0, 5.0, 0.0)

    if st.button("Check for Fraud", type="primary"):
        try:
            fraud_df = pd.read_csv('creditcard.csv')
            fraud_sample = fraud_df.sample(n=50000, random_state=42)
            X_f = fraud_sample.drop('Class', axis=1)
            y_f = fraud_sample['Class']
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
                X_f, y_f, test_size=0.2, random_state=42)
            sm_f = SMOTE(random_state=42)
            X_train_f, y_train_f = sm_f.fit_resample(X_train_f, y_train_f)
            sc_f = StandardScaler()
            X_train_f = sc_f.fit_transform(X_train_f)
            X_test_f = sc_f.transform(X_test_f)
            model_f = XGBClassifier(n_estimators=100, random_state=42,
                use_label_encoder=False, eval_metric='logloss')
            model_f.fit(X_train_f, y_train_f)

            input_f = X_test_f[0].reshape(1, -1)
            prob_f = model_f.predict_proba(input_f)[0][1]
            fraud_result = "🚨 FRAUD ALERT" if prob_f > 0.5 \
                else "✅ LEGITIMATE"

            st.markdown("---")
            st.subheader("Detection Result")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Transaction Status", fraud_result)
            with col_b:
                st.metric("Fraud Probability", f"{prob_f*100:.1f}%")

            if prob_f > 0.5:
                st.error(
                    "⚠️ This transaction has been flagged for review")
            else:
                st.success("Transaction cleared for processing")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure creditcard.csv is in the same folder")
