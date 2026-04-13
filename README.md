# 🏦 Credit Risk & Fraud Detection Engine

> End-to-end ML system detecting loan default risk and fraudulent transactions — deployed, explainable, and production-ready.

**[🚀 Live Demo](https://banking-ml-credit-fraud-3d6hyp2gp4y7xmu8ruj3jv.streamlit.app/)** &nbsp;·&nbsp; [LinkedIn](https://linkedin.com/in/gayathrikammidi) &nbsp;·&nbsp; [GitHub](https://github.com/gkammidi-prog)

---

## The Problem

Banks lose billions annually to loan defaults and payment fraud. Standard models either miss too much — or offer no explanation when they do flag something. This system solves both problems with a single, explainable, deployed ML engine.

---

## Results

| Model | AUC-ROC | Key Metric |
|-------|---------|------------|
| Credit Risk (XGBoost) | 0.639 | Precision-recall balanced |
| Fraud Detection (XGBoost) | **0.869** | **Recall: 75%** |

> A model that predicts "no fraud" on every transaction scores 99.8% accuracy — and catches zero fraud. **Recall** is the only metric that matters here.

---

## What This System Does

- Predicts **loan default probability** from applicant financial profile
- Flags **fraudulent transactions** across 284,807 records (0.17% fraud rate)
- Explains every prediction using **SHAP feature attribution** — not a black box
- Deployed as a **live interactive Streamlit dashboard** with 3 tabs

---

## Live Dashboard

```
Tab 1 — Credit Risk Predictor
  Input : age, loan amount, duration, housing, savings, employment
  Output: risk score + probability + SHAP waterfall chart

Tab 2 — Fraud Detector
  Input : transaction amount, time, anonymised risk features (V1–V10)
  Output: fraud flag + probability + SHAP waterfall chart

Tab 3 — Model Results Summary
  Side-by-side metrics, engineering decisions, audit-ready output
```

👉 **[Try it live → banking-ml-credit-fraud.streamlit.app](https://banking-ml-credit-fraud-3d6hyp2gp4y7xmu8ruj3jv.streamlit.app/)**

---

## Engineering Decisions

**SMOTE over undersampling**
Fraud is 0.17% of the dataset — extreme class imbalance. Undersampling destroys 99%+ of legitimate signal. SMOTE synthesises minority-class examples, preserving all available information while giving the model enough fraud cases to learn real patterns.

**Recall over accuracy**
On imbalanced data, accuracy is a misleading metric by design. The fraud model is explicitly optimised to minimise false negatives. In financial crime, one missed fraud always costs more — financially and reputationally — than a false alarm.

**SHAP over built-in feature importance**
XGBoost's native importance shows global averages. SHAP produces per-prediction attributions — showing exactly how much each feature pushed the score up or down for a specific transaction or applicant. This is the standard for explainability in regulated industries.

**XGBoost over baseline models**
Four algorithms benchmarked: Logistic Regression, Random Forest, Gradient Boosting, XGBoost. XGBoost outperformed all others on AUC-ROC and recall — its sequential error-correction makes it the strongest performer on structured tabular data.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Modeling | XGBoost · scikit-learn · Logistic Regression · Random Forest |
| Imbalance | SMOTE (imbalanced-learn) |
| Explainability | SHAP — per-prediction waterfall plots |
| Data | Pandas · NumPy |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
banking-ml-credit-fraud/
├── streamlit_app.py     # Live dashboard — 3 tabs, SHAP, cached models
├── credit_risk.py       # Standalone model training pipeline
├── requirements.txt     # Pinned dependencies
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/gkammidi-prog/banking-ml-credit-fraud
cd banking-ml-credit-fraud
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No local datasets needed — the app generates data at runtime.

---

## Datasets

| Dataset | Size | Source |
|---------|------|--------|
| German Credit Risk | 1,000 applicants · 10 features | UCI ML Repository / Kaggle (kabure) |
| Credit Card Fraud | 284,807 transactions · 30 features | Kaggle (mlg-ulb) |

---

## Portfolio

| Project | Domain | Highlight |
|---------|--------|-----------|
| **Banking ML** *(this repo)* | Credit & Fraud | AUC 0.869 · Fraud Recall 75% |
| Medicare HCC Risk Score | Healthcare | Recall 90.5% · 71,518 patient encounters |
| Hospital Readmission Predictor | Healthcare | SMOTE · XGBoost · Live dashboard |

---

## Author

**Gayathri Kammidi**  
MS Computer Science · Governors State University · May 2026  
4+ years in Data Engineering & ML — GCP · BigQuery · Airflow · Python · XGBoost

  [GitHub](https://github.com/gkammidi-prog)
