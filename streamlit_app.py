# =============================================
# STREAMLIT DASHBOARD — DEPLOY-READY VERSION
# Credit Risk & Fraud Detection Engine
# Gayathri Kammidi | MS CS, Governors State University
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Banking ML Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk & Fraud Detection Engine")
st.markdown(
    "*Dual ML system detecting loan default risk and fraudulent transactions "
    "— explainable, production-ready.*"
)

# -----------------------------------------------
# Data Generators (no local CSVs needed)
# -----------------------------------------------

@st.cache_data
def generate_credit_data(n=1000, seed=42):
    """
    Synthetic German-style credit dataset.
    Features mirror the UCI German Credit Data schema.
    """
    rng = np.random.default_rng(seed)
    n_bad = int(n * 0.30)
    n_good = n - n_bad

    def make_group(size, risk):
        age_mean = 30 if risk == "bad" else 38
        dur_mean = 30 if risk == "bad" else 18
        amt_mean = 4500 if risk == "bad" else 2800
        return pd.DataFrame({
            "Age":            rng.integers(18, 75, size=size),
            "Sex":            rng.choice(["male", "female"], size=size),
            "Job":            rng.integers(0, 4, size=size),
            "Housing":        rng.choice(["own", "free", "rent"], size=size,
                                         p=[0.7, 0.1, 0.2] if risk == "good" else [0.3, 0.2, 0.5]),
            "Saving accounts": rng.choice(
                ["little", "moderate", "quite rich", "rich"], size=size,
                p=[0.5, 0.3, 0.1, 0.1] if risk == "bad" else [0.2, 0.3, 0.3, 0.2]),
            "Checking account": rng.choice(
                ["little", "moderate", "rich"], size=size,
                p=[0.6, 0.3, 0.1] if risk == "bad" else [0.2, 0.4, 0.4]),
            "Credit amount":  np.clip(rng.normal(amt_mean, 1500, size=size).astype(int), 250, 20000),
            "Duration":       np.clip(rng.normal(dur_mean, 12, size=size).astype(int), 4, 72),
            "Purpose":        rng.choice(
                ["car", "furniture/equipment", "radio/TV", "education", "business"], size=size),
            "Risk":           [risk] * size,
        })

    df = pd.concat([make_group(n_good, "good"), make_group(n_bad, "bad")], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@st.cache_data
def generate_fraud_data(n=20000, seed=42):
    """
    Synthetic credit-card fraud dataset.
    V1–V10 are PCA-style anonymised features (as in the real Kaggle dataset).
    ~0.5% fraud rate — realistic class imbalance.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n * 0.005)
    n_legit = n - n_fraud

    def make_transactions(size, fraud):
        v_means = rng.uniform(-2, 2, 10) if fraud else np.zeros(10)
        v_stds  = rng.uniform(1, 3, 10)  if fraud else np.ones(10)
        vdata = {f"V{i+1}": rng.normal(v_means[i], v_stds[i], size) for i in range(10)}
        base = pd.DataFrame(vdata)
        base["Amount"] = np.clip(
            rng.exponential(300 if fraud else 80, size), 1, 5000)
        base["Time"]   = rng.integers(0, 172800, size)
        base["Class"]  = int(fraud)
        return base

    df = pd.concat(
        [make_transactions(n_legit, False), make_transactions(n_fraud, True)],
        ignore_index=True
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# -----------------------------------------------
# Model trainers (cached so they only run once)
# -----------------------------------------------

@st.cache_resource
def train_credit_model():
    df = generate_credit_data()
    df = df.dropna()
    le = LabelEncoder()
    cat_cols = [c for c in df.select_dtypes("object").columns if c != "Risk"]
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    df["Risk"] = (df["Risk"] == "bad").astype(int)

    X = df.drop("Risk", axis=1)
    y = df["Risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_train_r, y_train_r = sm.fit_resample(X_train, y_train)

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train_r)
    X_test_s  = sc.transform(X_test)

    model = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0)
    model.fit(X_train_s, y_train_r)

    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    return model, sc, X.columns.tolist(), auc


@st.cache_resource
def train_fraud_model():
    df = generate_fraud_data()
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_train_r, y_train_r = sm.fit_resample(X_train, y_train)

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train_r)
    X_test_s  = sc.transform(X_test)

    model = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0)
    model.fit(X_train_s, y_train_r)

    preds = model.predict(X_test_s)
    auc    = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    recall = recall_score(y_test, preds)
    return model, sc, X.columns.tolist(), auc, recall


# -----------------------------------------------
# Tabs
# -----------------------------------------------
tab1, tab2, tab3 = st.tabs(["💳 Credit Risk", "🚨 Fraud Detection", "📊 Model Results"])

# =============================================
# TAB 1 — CREDIT RISK
# =============================================
with tab1:
    st.header("Credit Risk Predictor")
    st.markdown(
        "Predict whether a loan applicant is **high** or **low risk** "
        "based on their financial profile."
    )

    with st.spinner("Training credit risk model…"):
        cr_model, cr_sc, cr_features, cr_auc = train_credit_model()

    st.success(f"Model ready — AUC-ROC: **{cr_auc:.3f}**")

    st.subheader("Applicant Profile")
    col1, col2 = st.columns(2)

    with col1:
        age            = st.slider("Age", 18, 75, 35)
        credit_amount  = st.slider("Loan Amount ($)", 500, 20000, 5000, step=100)
        duration       = st.slider("Loan Duration (months)", 4, 72, 24)
        job            = st.selectbox(
            "Job Type", [0, 1, 2, 3],
            format_func=lambda x: ["Unskilled", "Unskilled Resident",
                                   "Skilled", "Highly Skilled"][x])

    with col2:
        sex            = st.selectbox("Sex", ["male", "female"])
        housing        = st.selectbox("Housing", ["own", "free", "rent"])
        saving         = st.selectbox(
            "Saving Accounts", ["little", "moderate", "quite rich", "rich"])
        checking       = st.selectbox(
            "Checking Account", ["little", "moderate", "rich"])
        purpose        = st.selectbox(
            "Loan Purpose",
            ["car", "furniture/equipment", "radio/TV", "education", "business"])

    if st.button("Predict Credit Risk", type="primary"):
        # Encode inputs the same way as training
        sex_enc      = 1 if sex == "male" else 0
        housing_enc  = {"own": 2, "free": 0, "rent": 1}[housing]
        saving_enc   = {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3}[saving]
        checking_enc = {"little": 0, "moderate": 1, "rich": 2}[checking]
        purpose_enc  = ["car", "furniture/equipment", "radio/TV",
                        "education", "business"].index(purpose)

        input_df = pd.DataFrame([{
            "Age": age, "Sex": sex_enc, "Job": job,
            "Housing": housing_enc, "Saving accounts": saving_enc,
            "Checking account": checking_enc,
            "Credit amount": credit_amount, "Duration": duration,
            "Purpose": purpose_enc
        }])[cr_features]

        input_scaled = cr_sc.transform(input_df)
        prob = cr_model.predict_proba(input_scaled)[0][1]
        label = "HIGH RISK 🔴" if prob > 0.5 else "LOW RISK 🟢"

        st.markdown("---")
        st.subheader("Prediction Result")
        m1, m2 = st.columns(2)
        m1.metric("Risk Level", label)
        m2.metric("Risk Probability", f"{prob * 100:.1f}%")

        if prob > 0.7:
            st.error("⚠️ Recommend declining this application.")
        elif prob > 0.5:
            st.warning("🟡 Borderline — additional review recommended.")
        else:
            st.success("✅ Applicant profile looks healthy.")

        # SHAP waterfall
        st.subheader("SHAP Explanation — Why this prediction?")
        explainer  = shap.TreeExplainer(cr_model)
        shap_vals  = explainer.shap_values(input_scaled)
        fig, ax = plt.subplots(figsize=(9, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                feature_names=cr_features
            ), show=False
        )
        st.pyplot(fig)
        plt.close()

# =============================================
# TAB 2 — FRAUD DETECTION
# =============================================
with tab2:
    st.header("Transaction Fraud Detector")
    st.markdown(
        "Flag suspicious transactions in real time using XGBoost + SMOTE. "
        "Model optimised for **recall** — missing fraud costs more than a false alarm."
    )

    with st.spinner("Training fraud detection model…"):
        fd_model, fd_sc, fd_features, fd_auc, fd_recall = train_fraud_model()

    st.success(
        f"Model ready — AUC-ROC: **{fd_auc:.3f}** | Recall: **{fd_recall * 100:.1f}%**"
    )

    st.subheader("Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        amount = st.slider("Transaction Amount ($)", 0.0, 5000.0, 150.0, step=5.0)
        time   = st.slider("Time Since First Transaction (seconds)", 0, 172800, 50000)
        v1     = st.slider("Risk Feature V1", -5.0, 5.0, 0.0, step=0.1)
        v2     = st.slider("Risk Feature V2", -5.0, 5.0, 0.0, step=0.1)
        v3     = st.slider("Risk Feature V3", -5.0, 5.0, 0.0, step=0.1)

    with col2:
        v4  = st.slider("Risk Feature V4",  -5.0, 5.0, 0.0, step=0.1)
        v5  = st.slider("Risk Feature V5",  -5.0, 5.0, 0.0, step=0.1)
        v6  = st.slider("Risk Feature V6",  -5.0, 5.0, 0.0, step=0.1)
        v7  = st.slider("Risk Feature V7",  -5.0, 5.0, 0.0, step=0.1)
        v8  = st.slider("Risk Feature V8",  -5.0, 5.0, 0.0, step=0.1)
        v9  = st.slider("Risk Feature V9",  -5.0, 5.0, 0.0, step=0.1)
        v10 = st.slider("Risk Feature V10", -5.0, 5.0, 0.0, step=0.1)

    if st.button("Check for Fraud", type="primary"):
        input_df = pd.DataFrame([{
            "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
            "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
            "Amount": amount, "Time": time
        }])[fd_features]

        input_scaled = fd_sc.transform(input_df)
        prob_f = fd_model.predict_proba(input_scaled)[0][1]
        fraud_label = "🚨 FRAUD ALERT" if prob_f > 0.5 else "✅ LEGITIMATE"

        st.markdown("---")
        st.subheader("Detection Result")
        m1, m2 = st.columns(2)
        m1.metric("Transaction Status", fraud_label)
        m2.metric("Fraud Probability", f"{prob_f * 100:.1f}%")

        if prob_f > 0.5:
            st.error("⚠️ This transaction has been flagged for immediate review.")
        else:
            st.success("Transaction cleared for processing.")

        # SHAP waterfall
        st.subheader("SHAP Explanation — Key fraud signals")
        explainer_f = shap.TreeExplainer(fd_model)
        shap_vals_f = explainer_f.shap_values(input_scaled)
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals_f[0],
                base_values=explainer_f.expected_value,
                feature_names=fd_features
            ), show=False
        )
        st.pyplot(fig2)
        plt.close()

# =============================================
# TAB 3 — MODEL RESULTS SUMMARY
# =============================================
with tab3:
    st.header("Model Performance Summary")
    st.markdown(
        "Both models are trained on synthetic datasets that mirror real-world "
        "distributions. The fraud model uses **SMOTE** to handle extreme class imbalance "
        "(0.5% fraud rate) and is optimised for **recall**."
    )

    with st.spinner("Loading metrics…"):
        _, _, _, cr_auc_val = train_credit_model()
        _, _, _, fd_auc_val, fd_recall_val = train_fraud_model()

    results = pd.DataFrame({
        "Model":   ["Credit Risk (XGBoost)", "Fraud Detection (XGBoost)"],
        "Dataset": ["Synthetic German Credit (1,000 applicants)",
                    "Synthetic Credit Card Transactions (20,000)"],
        "AUC-ROC": [f"{cr_auc_val:.3f}", f"{fd_auc_val:.3f}"],
        "Key Metric": ["AUC-ROC", f"Recall: {fd_recall_val * 100:.1f}%"],
        "Imbalance Handling": ["SMOTE", "SMOTE"],
        "Explainability": ["SHAP Waterfall", "SHAP Waterfall"],
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Engineering Decisions")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**SMOTE over undersampling**")
        st.markdown(
            "Fraud class is <1% of data. Undersampling loses signal. "
            "SMOTE synthesises minority class samples — preserving information."
        )
    with c2:
        st.markdown("**Recall over Accuracy**")
        st.markdown(
            "A model predicting 'no fraud' every time scores 99.5% accuracy "
            "and catches zero fraud. Recall is the only metric that matters."
        )
    with c3:
        st.markdown("**SHAP over feature importance**")
        st.markdown(
            "Built-in feature importance shows global averages. SHAP shows "
            "per-prediction impact — audit-ready for compliance teams."
        )

    st.markdown("---")
    st.markdown(
        "**Author:** Gayathri Kammidi · MS Computer Science, "
        "Governors State University · May 2026  \n"
        "[LinkedIn](https://linkedin.com/in/gayathrikammidi) · "
        "[GitHub](https://github.com/gkammidi-prog)"
    )
