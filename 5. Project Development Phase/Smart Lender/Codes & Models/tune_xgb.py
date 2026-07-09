"""
tune_xgb.py — XGBoost Hyperparameter Tuning Script
Smart Lender | Applicant Credibility Prediction

This script performs standalone XGBoost hyperparameter tuning using
RandomizedSearchCV, then saves the best-tuned model to rdf.pkl for
deployment via the Flask API (app.py).

Pipeline:
    1. Load and preprocess the loan prediction dataset
    2. Apply label encoding + log transformation
    3. Stratified 80-20 train-test split
    4. SMOTE oversampling for class balance
    5. StandardScaler normalization
    6. RandomizedSearchCV for XGBoost tuning
    7. Evaluate: accuracy, classification report, confusion matrix, F1
    8. Save tuned model bundle to rdf.pkl
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Use fivethirtyeight style for consistent and readable plots
plt.style.use('fivethirtyeight')

PLOT_DIR = 'static/images/plots'
os.makedirs(PLOT_DIR, exist_ok=True)


# ─── 1. Load Dataset ──────────────────────────────────────────────────────────
print("=" * 55)
print("  Smart Lender — XGBoost Hyperparameter Tuning")
print("=" * 55)

data = pd.read_csv('Dataset/loan_prediction.csv')
print(f"\n[1] Dataset loaded: {data.shape[0]} rows x {data.shape[1]} columns")

# ─── 2. Preprocessing ─────────────────────────────────────────────────────────
print("\n[2] Handling Categorical values (mapping)...")
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

print("\n[3] Handling Missing values & Type Casting...")
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])

data['Dependents'] = data['Dependents'].astype(str).str.replace('+', '', regex=False)
data['Dependents'] = data['Dependents'].replace('nan', np.nan)
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])

data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mode()[0])

data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['CoapplicantIncome'] = data['CoapplicantIncome'].fillna(data['CoapplicantIncome'].mean())
data['ApplicantIncome'] = data['ApplicantIncome'].fillna(data['ApplicantIncome'].mean())

# Changing the datatype of each float/object column to int64
data['Gender'] = data['Gender'].astype('int64')
data['Married'] = data['Married'].astype('int64')
data['Dependents'] = data['Dependents'].astype('int64')
data['Self_Employed'] = data['Self_Employed'].astype('int64')
data['CoapplicantIncome'] = data['CoapplicantIncome'].astype('int64')
data['LoanAmount'] = data['LoanAmount'].astype('int64')
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype('int64')
data['Credit_History'] = data['Credit_History'].astype('int64')

# ─── 4. Feature Selection ─────────────────────────────────────────────────────
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area']

X = data[features]
y = data['Loan_Status']
print(f"\n[4] Features: {features}")
print(f"    Target distribution before SMOTE: {dict(y.value_counts())}")

# ─── 5. SMOTE Oversampling ────────────────────────────────────────────────────
print("\n[5] Applying SMOTE oversampling to balance target classes...")
smote = SMOTE(random_state=42)
x_bal, y_bal = smote.fit_resample(X, y)
print(f"    After SMOTE: {dict(y_bal.value_counts())}")
names = x_bal.columns

# ─── 6. Feature Scaling ───────────────────────────────────────────────────────
print("\n[6] Applying StandardScaler normalization on balanced features...")
scaler = StandardScaler()
x_bal_scaled = scaler.fit_transform(x_bal)
x_bal_scaled_df = pd.DataFrame(x_bal_scaled, columns=names)

# ─── 7. Train-Test Split (67-33) ──────────────────────────────────────────────
# splitting the dataset in train and test on balanced dataset
X_train, X_test, y_train, y_test = train_test_split(
    x_bal_scaled_df, y_bal, test_size=0.33, random_state=42
)
print(f"\n[7] Split complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# ─── 8. RandomizedSearchCV — XGBoost Tuning ──────────────────────────────────
print("\n" + "=" * 55)
print("  RandomizedSearchCV — XGBoost Hyperparameter Tuning")
print("=" * 55)

# Hyperparameter search space
param_distributions = {
    'n_estimators':     [50, 75, 100, 125],
    'max_depth':        [2, 3, 4, 5],
    'learning_rate':    [0.05, 0.1, 0.15, 0.2],
    'subsample':        [0.6, 0.7, 0.8],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'gamma':            [0.1, 0.2, 0.3, 0.5],
    'min_child_weight': [3, 5, 7, 10],
    'reg_alpha':        [0.1, 0.5, 1.0, 2.0],
    'reg_lambda':       [1.0, 2.0, 3.0, 5.0],
}

xgb_base = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=30,           # Number of random parameter combinations to try
    cv=5,                # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nFitting RandomizedSearchCV (30 iterations x 5-fold CV)...")
search.fit(X_train, y_train)

print(f"\nBest CV Score:  {search.best_score_*100:.2f}%")
print(f"Best Params:")
for k, v in search.best_params_.items():
    print(f"    {k}: {v}")

# ─── 9. Final Model Evaluation ────────────────────────────────────────────────
best_xgb = search.best_estimator_

y_train_pred = best_xgb.predict(X_train)
y_test_pred  = best_xgb.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)
f1        = f1_score(y_test, y_test_pred, average='macro')

print("\n" + "=" * 55)
print("  Final Tuned XGBoost — Evaluation Results")
print("=" * 55)
print(f"  Training Accuracy : {train_acc * 100:.2f}%")
print(f"  Testing  Accuracy : {test_acc  * 100:.2f}%")
print(f"  F1-Score (macro)  : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Rejected', 'Approved']))

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)

# ─── 10. Plot: Confusion Matrix ───────────────────────────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
plt.title('XGBoost (Tuned) Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"\nConfusion matrix saved to {PLOT_DIR}/confusion_matrix.png")

# ─── 11. Plot: Cross-Validation Score Distribution ────────────────────────────
cv_results = pd.DataFrame(search.cv_results_)
top_n = 10
top_results = cv_results.nlargest(top_n, 'mean_test_score')

plt.figure(figsize=(10, 5))
plt.barh(
    range(top_n),
    top_results['mean_test_score'] * 100,
    xerr=top_results['std_test_score'] * 100,
    color='#8a2be2', alpha=0.8, capsize=5
)
plt.yticks(range(top_n), [f'Config {i+1}' for i in range(top_n)])
plt.xlabel('CV Accuracy (%)')
plt.title('Top 10 XGBoost Configurations (RandomizedSearchCV)')
plt.xlim(60, 100)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/xgb_cv_scores.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"CV scores chart saved to {PLOT_DIR}/xgb_cv_scores.png")

# ─── 12. Save Tuned Model Bundle ──────────────────────────────────────────────
print("\n[12] Saving tuned model to rdf.pkl...")

model_bundle = {
    'model':    best_xgb,
    'scaler':   scaler,
    'features': features,
    'metrics': {
        'xgb': {'train': train_acc, 'test': test_acc, 'f1': f1},
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_
    }
}

with open('rdf.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

with open('scale1.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("    Saved: rdf.pkl  (best XGBoost + scaler + features + metrics)")
print("    Saved: scale1.pkl  (scaler only)")

print("\n" + "=" * 55)
print("  Tuning Complete!")
print(f"  Best XGBoost -> Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")
print("=" * 55)

