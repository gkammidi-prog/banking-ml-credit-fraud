# =============================================
# PROJECT 1 - BANKING ML
# Credit Risk & Fraud Detection Engine
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report,
                             roc_auc_score,
                             confusion_matrix,
                             RocCurveDisplay)
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

print("All libraries loaded successfully!")

# =============================================
# PART 1 - CREDIT RISK MODEL
# =============================================

# --- Load Data ---
print("\nLoading Credit Risk Dataset...")
credit_df = pd.read_csv('german_credit_data.csv', index_col=0)
print(f"Shape: {credit_df.shape}")
print(credit_df.head())
print("\nColumn Names:", credit_df.columns.tolist())
print("\nNull Values:\n", credit_df.isnull().sum())
print("\nAll Column Names:", credit_df.columns.tolist())
print("\nLast Column (likely target):", credit_df.iloc[:, -1].value_counts())

# --- Clean Data ---
credit_df.dropna(inplace=True)

# --- Encode Categorical Columns ---
le = LabelEncoder()
cat_cols = credit_df.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != 'Risk']

for col in cat_cols:
    credit_df[col] = le.fit_transform(credit_df[col].astype(str))

credit_df['Risk'] = (credit_df['Risk'] == 'bad').astype(int)

# --- Features & Target ---
X_credit = credit_df.drop('Risk', axis=1)
y_credit = credit_df['Risk']

# --- Train Test Split ---
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42)

# --- Handle Imbalance with SMOTE ---
smote = SMOTE(random_state=42)
X_train_c, y_train_c = smote.fit_resample(X_train_c, y_train_c)

# --- Scale Features ---
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# --- Train XGBoost ---
print("\nTraining Credit Risk Model...")
xgb_credit = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_credit.fit(X_train_c, y_train_c)

# --- Evaluate ---
y_pred_c = xgb_credit.predict(X_test_c)
y_prob_c = xgb_credit.predict_proba(X_test_c)[:, 1]

print("\n=== CREDIT RISK RESULTS ===")
print(classification_report(y_test_c, y_pred_c))
print(f"AUC-ROC Score: {roc_auc_score(y_test_c, y_prob_c):.4f}")

# --- Save AUC Score for LinkedIn ---
auc_credit = roc_auc_score(y_test_c, y_prob_c)
print(f"\nYour LinkedIn metric: AUC-ROC = {auc_credit:.3f}")

# --- SHAP Explainability ---
print("\nGenerating SHAP explanations...")
explainer_credit = shap.TreeExplainer(xgb_credit)
shap_values_credit = explainer_credit.shap_values(X_test_c)

plt.figure()
shap.summary_plot(
    shap_values_credit,
    X_test_c,
    feature_names=X_credit.columns.tolist(),
    show=False
)
plt.tight_layout()
plt.savefig('shap_credit_risk.png', dpi=150, bbox_inches='tight')
print("SHAP plot saved as shap_credit_risk.png")

# =============================================
# PART 2 - FRAUD DETECTION MODEL
# =============================================

print("\nLoading Fraud Detection Dataset...")
fraud_df = pd.read_csv('creditcard.csv')
print(f"Shape: {fraud_df.shape}")
print(f"\nFraud Distribution:\n{fraud_df['Class'].value_counts()}")
print(f"Fraud %: {fraud_df['Class'].mean()*100:.3f}%")

# --- Sample for speed (use 50k rows) ---
fraud_sample = fraud_df.sample(n=50000, random_state=42)

# --- Features & Target ---
X_fraud = fraud_sample.drop('Class', axis=1)
y_fraud = fraud_sample['Class']

# --- Train Test Split ---
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fraud, y_fraud, test_size=0.2, random_state=42)

# --- Handle Imbalance with SMOTE ---
smote_f = SMOTE(random_state=42)
X_train_f, y_train_f = smote_f.fit_resample(X_train_f, y_train_f)

# --- Scale ---
scaler_f = StandardScaler()
X_train_f = scaler_f.fit_transform(X_train_f)
X_test_f = scaler_f.transform(X_test_f)

# --- Train XGBoost ---
print("\nTraining Fraud Detection Model...")
xgb_fraud = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_fraud.fit(X_train_f, y_train_f)

# --- Evaluate ---
y_pred_f = xgb_fraud.predict(X_test_f)
y_prob_f = xgb_fraud.predict_proba(X_test_f)[:, 1]

print("\n=== FRAUD DETECTION RESULTS ===")
print(classification_report(y_test_f, y_pred_f))
print(f"AUC-ROC Score: {roc_auc_score(y_test_f, y_prob_f):.4f}")

# --- Save Recall for LinkedIn ---
from sklearn.metrics import recall_score
recall_fraud = recall_score(y_test_f, y_pred_f)
print(f"\nYour LinkedIn metric: Recall = {recall_fraud*100:.1f}%")

# --- SHAP Explainability ---
print("\nGenerating SHAP explanations for fraud...")
explainer_fraud = shap.TreeExplainer(xgb_fraud)
shap_values_fraud = explainer_fraud.shap_values(X_test_f[:100])

plt.figure()
shap.summary_plot(
    shap_values_fraud,
    X_test_f[:100],
    feature_names=X_fraud.columns.tolist(),
    show=False
)
plt.tight_layout()
plt.savefig('shap_fraud.png', dpi=150, bbox_inches='tight')
print("SHAP plot saved as shap_fraud.png")

print("\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("Next step: Run streamlit_app.py to launch your dashboard")
