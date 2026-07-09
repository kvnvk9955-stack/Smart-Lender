import os
import sys
# Add local user site-packages to sys.path
user_site = os.path.expanduser('~/.local/lib/python3.10/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from imblearn.over_sampling import SMOTE

# ─── Visualization Style ──────────────────────────────────────────────────────
# Use fivethirtyeight style for consistent and readable plots (as per project spec)
plt.style.use('fivethirtyeight')

script_dir = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(script_dir, 'static', 'images', 'plots')


def main():
    print("=" * 60)
    print("   Smart Lender — Machine Learning Pipeline")
    print("=" * 60)

    # Ensure plots directory exists
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ─── STEP 1: Read the Dataset ──────────────────────────────────────────────
    # The dataset is in CSV format; read using pandas read_csv()
    dataset_path = os.path.join(script_dir, 'Dataset', 'loan_prediction.csv')
    data = pd.read_csv(dataset_path)
    print(f"\n[1] Dataset loaded successfully.")
    print(f"    Shape: {data.shape}  (rows x columns)")
    print(f"\n    First 5 rows:")
    print(data.head().to_string())
    print(f"\n    Columns: {list(data.columns)}")
    print(f"    Missing values per column:\n{data.isnull().sum().to_string()}")

    # ─── STEP 2: Univariate Analysis ──────────────────────────────────────────
    print("\n[2] Generating Univariate EDA plots...")
    try:
        # ── 2a. Distribution Plots (Distplot style) for continuous variables ──
        # ApplicantIncome — right-skewed distribution
        # Credit_History  — binary feature with values 0 and 1
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(data['ApplicantIncome'], color='crimson', kde=True, ax=axes[0])
        axes[0].set_title('Applicant Income Distribution')
        axes[0].set_xlabel('Applicant Income')
        axes[0].set_ylabel('Density')

        credit_clean = data['Credit_History'].dropna()
        sns.histplot(credit_clean, kde=True, ax=axes[1], color='steelblue')
        axes[1].set_title('Credit History Distribution')
        axes[1].set_xlabel('Credit History (0=Poor, 1=Good)')
        axes[1].set_ylabel('Density')

        plt.suptitle('Univariate Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/univariate_dist.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("    Saved: univariate_dist.png")

        # ── 2b. Count Plots for categorical variables (1x4 subplot layout) ──
        # Gender and Education: two-category features
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        sns.countplot(x='Gender', data=data, hue='Gender', legend=False, ax=axes[0],
                      palette={'Male': '#8a2be2', 'Female': '#00e5ff'})
        axes[0].set_title('Gender Distribution')
        axes[0].set_xlabel('Gender')

        sns.countplot(x='Education', data=data, hue='Education', legend=False, ax=axes[1],
                      palette={'Graduate': '#8a2be2', 'Not Graduate': '#00e5ff'})
        axes[1].set_title('Education Distribution')
        axes[1].set_xlabel('Education')

        sns.countplot(x='Married', data=data, hue='Married', legend=False, ax=axes[2],
                      palette={'Yes': '#8a2be2', 'No': '#00e5ff'})
        axes[2].set_title('Marital Status')
        axes[2].set_xlabel('Married')

        sns.countplot(x='Loan_Status', data=data, hue='Loan_Status', legend=False, ax=axes[3],
                      palette={'N': '#ef4444', 'Y': '#22c55e'})
        axes[3].set_title('Loan Status (Target)')
        axes[3].set_xlabel('Loan Status (Y/N)')

        plt.suptitle('Categorical Feature Count Plots', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/univariate_count.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("    Saved: univariate_count.png")

        # Individual plots used by home.html dashboard
        plt.figure(figsize=(7, 4))
        sns.histplot(data['ApplicantIncome'], color='crimson', kde=True)
        plt.title('Applicant Income — KDE Distribution')
        plt.xlabel('Applicant Income')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/income_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(5, 4))
        sns.countplot(x='Gender', data=data, hue='Gender', legend=False,
                      palette={'Male': '#8a2be2', 'Female': '#00e5ff'})
        plt.title('Gender Distribution')
        plt.xlabel('Gender')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/gender_count.png', dpi=100, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(5, 4))
        sns.countplot(x='Loan_Status', data=data, hue='Loan_Status', legend=False,
                      palette={'N': '#ef4444', 'Y': '#22c55e'})
        plt.title('Loan Approval Status Count')
        plt.xlabel('Loan Status')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/loan_status_count.png', dpi=100, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(5, 4))
        sns.countplot(x='Education', hue='Loan_Status', data=data,
                      palette=['#ef4444', '#22c55e'])
        plt.title('Loan Status by Education')
        plt.xlabel('Education')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/education_vs_loan_status.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("    Saved: income_distribution, gender_count, loan_status_count, education_vs_loan_status")
    except Exception as plot_err:
        print(f"    Warning: Skipping Step 2 Univariate plots due to: {plot_err}")

    # ─── STEP 3: Bivariate Analysis ───────────────────────────────────────────
    print("\n[3] Generating Bivariate EDA plots...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Insight 1: Married vs Gender
        sns.countplot(x='Married', hue='Gender', data=data,
                      palette=['#8a2be2', '#00e5ff'], ax=axes[0])
        axes[0].set_title('Marital Status vs Gender')
        axes[0].set_xlabel('Married')
        axes[0].legend(title='Gender')

        # Insight 2: Self_Employed vs Education
        sns.countplot(x='Self_Employed', hue='Education', data=data,
                      palette=['#f59e0b', '#10b981'], ax=axes[1])
        axes[1].set_title('Self-Employment vs Education')
        axes[1].set_xlabel('Self Employed')
        axes[1].legend(title='Education')

        # Insight 3: Property Area vs Loan Amount Term (top 4 terms for clarity)
        top_terms = data['Loan_Amount_Term'].value_counts().nlargest(4).index
        biv_data = data[data['Loan_Amount_Term'].isin(top_terms)].copy()
        biv_data['Loan_Amount_Term'] = biv_data['Loan_Amount_Term'].astype(int).astype(str)
        sns.countplot(x='Property_Area', hue='Loan_Amount_Term', data=biv_data,
                      palette='viridis', ax=axes[2])
        axes[2].set_title('Property Area vs Loan Amount Term')
        axes[2].set_xlabel('Property Area')
        axes[2].legend(title='Loan Term (Days)', fontsize=8)

        plt.suptitle('Bivariate Analysis — Feature Relationships', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/bivariate.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("    Saved: bivariate.png")
    except Exception as plot_err:
        print(f"    Warning: Skipping Step 3 Bivariate plots due to: {plot_err}")

    print("\n    EDA Observations:")
    print("    * ApplicantIncome exhibits a right-skewed (long-tail) distribution.")
    print("    * Credit_History is a binary feature with values 0 and 1.")
    print("    * Gender and Education are categorical variables with two categories each.")
    print("    * Frequency of 'Male' and 'Graduate' is higher than respective counterparts.")
    print("    * Married males represent the largest group in loan applications.")
    print("    * Educated applicants tend to be salaried (not self-employed).")
    print("    * 360-day loan term dominates across all property areas.")

    # ─── STEP 4: Handling Categorical Values ──────────────────────────────────
    print("\n[4] Handling Categorical values (mapping)...")
    data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
    data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
    data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
    data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
    print("    Categorical columns mapped.")

    # ─── STEP 5: Handling Missing Values & Type Casting ──────────────────────
    print("\n[5] Handling Missing values & Type Casting...")
    # Missing values in numerical features are replaced with the mean of the respective column.
    # Missing values in categorical features are replaced with the mode (most frequent value).
    # LoanAmount is filled with mode as shown in the notebook.
    
    data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
    data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
    
    # Replacing '+' with '' in Dependents
    data['Dependents'] = data['Dependents'].astype(str).str.replace('+', '', regex=False)
    # Handle 'nan' strings if any
    data['Dependents'] = data['Dependents'].replace('nan', np.nan)
    data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
    
    data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
    data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mode()[0])
    
    # Fill remaining columns: Credit_History (mode), Loan_Amount_Term (mode), CoapplicantIncome/ApplicantIncome (mean/mode)
    data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
    data['CoapplicantIncome'] = data['CoapplicantIncome'].fillna(data['CoapplicantIncome'].mean())
    data['ApplicantIncome'] = data['ApplicantIncome'].fillna(data['ApplicantIncome'].mean())

    # Changing the datatype of each float/object column to int64 as shown in the notebook
    data['Gender'] = data['Gender'].astype('int64')
    data['Married'] = data['Married'].astype('int64')
    data['Dependents'] = data['Dependents'].astype('int64')
    data['Self_Employed'] = data['Self_Employed'].astype('int64')
    data['CoapplicantIncome'] = data['CoapplicantIncome'].astype('int64')
    data['LoanAmount'] = data['LoanAmount'].astype('int64')
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype('int64')
    data['Credit_History'] = data['Credit_History'].astype('int64')
    print("    Missing values filled and columns cast to int64.")
    print(data.info())

    # ─── STEP 6: Multivariate Swarm Plot ──────────────────────────────────────
    print("\n[6] Generating Swarm Plot (Multivariate analysis)...")
    try:
        plt.figure(figsize=(10, 6))
        # visualized based on gender and income what would be the application status
        sns.swarmplot(x='Gender', y='ApplicantIncome', hue='Loan_Status', data=data, palette=['#ef4444', '#22c55e'])
        plt.title('Loan Application Status by Gender and Applicant Income')
        plt.xlabel('Gender (0 = Male, 1 = Female)')
        plt.ylabel('Applicant Income')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/multivariate.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("    Saved: multivariate.png")
    except Exception as plot_err:
        print(f"    Warning: Skipping Step 6 Swarm Plot due to: {plot_err}")

    # ─── STEP 7: X and y Selection ────────────────────────────────────────────
    features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    X = data[features]
    y = data['Loan_Status']
    print(f"\n[7] Target distribution before balancing:\n{y.value_counts().to_string()}")

    # ─── STEP 8: SMOTE Oversampling ───────────────────────────────────────────
    print("\n[8] Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    x_bal, y_bal = smote.fit_resample(X, y)
    print("    Target distribution after SMOTE:")
    print(y_bal.value_counts().to_string())
    names = x_bal.columns

    # ─── STEP 9: Feature Scaling ─────────────────────────────────────────────
    print("\n[9] Applying StandardScaler normalization on balanced features...")
    sc = StandardScaler()
    x_bal_scaled = sc.fit_transform(x_bal)
    x_bal_scaled_df = pd.DataFrame(x_bal_scaled, columns=names)

    # ─── STEP 10: Train-Test Split (67-33) ────────────────────────────────────
    # splitting the dataset in train and test on balanced dataset
    X_train, X_test, y_train, y_test = train_test_split(
        x_bal_scaled_df, y_bal, test_size=0.33, random_state=42
    )
    print(f"\n[10] Split complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # ─── STEP 11: Model Training ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   Model Training & Accuracy Comparison")
    print("=" * 60)

    results = {}

    # A. Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
    dt_test_acc  = accuracy_score(y_test,  dt.predict(X_test))
    results['dt'] = {'train': dt_train_acc, 'test': dt_test_acc}
    print(f"\n  Decision Tree     | Train: {dt_train_acc*100:.2f}% | Test: {dt_test_acc*100:.2f}%")

    # B. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
    rf_test_acc  = accuracy_score(y_test,  rf.predict(X_test))
    results['rf'] = {'train': rf_train_acc, 'test': rf_test_acc}
    print(f"  Random Forest     | Train: {rf_train_acc*100:.2f}% | Test: {rf_test_acc*100:.2f}%")

    # C. KNN
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
    knn_test_acc  = accuracy_score(y_test,  knn.predict(X_test))
    results['knn'] = {'train': knn_train_acc, 'test': knn_test_acc}
    print(f"  KNN (k=7)         | Train: {knn_train_acc*100:.2f}% | Test: {knn_test_acc*100:.2f}%")

    # D. Gradient Boosting
    gb = GradientBoostingClassifier(max_depth=5, n_estimators=50,
                                     learning_rate=0.2, random_state=42)
    gb.fit(X_train, y_train)
    gb_train_acc = accuracy_score(y_train, gb.predict(X_train))
    gb_test_acc  = accuracy_score(y_test,  gb.predict(X_test))
    results['gb'] = {'train': gb_train_acc, 'test': gb_test_acc}
    print(f"  Gradient Boost    | Train: {gb_train_acc*100:.2f}% | Test: {gb_test_acc*100:.2f}%")

    # E. XGBoost with RandomizedSearchCV
    print("\n  [XGBoost — RandomizedSearchCV Tuning in progress...]")
    from xgboost import XGBClassifier
    xgb_params = {
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
    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_params,
        n_iter=25, cv=5, scoring='accuracy',
        n_jobs=-1, random_state=42, verbose=0
    )
    xgb_search.fit(X_train, y_train)
    print(f"  Best params: {xgb_search.best_params_}")

    xgb = xgb_search.best_estimator_
    xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train))
    xgb_test_acc  = accuracy_score(y_test,  xgb.predict(X_test))
    results['xgb'] = {'train': xgb_train_acc, 'test': xgb_test_acc}
    print(f"  XGBoost (Tuned)   | Train: {xgb_train_acc*100:.2f}% | Test: {xgb_test_acc*100:.2f}%")

    # ─── STEP 12: XGBoost Full Evaluation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("   XGBoost Classification Report")
    print("=" * 60)
    xgb_preds = xgb.predict(X_test)
    print(classification_report(y_test, xgb_preds, target_names=['Rejected', 'Approved']))
    cm = confusion_matrix(y_test, xgb_preds)
    print("Confusion Matrix:")
    print(cm)
    print(f"F1-Score (macro): {f1_score(y_test, xgb_preds, average='macro'):.4f}")

    # ─── STEP 13: Confusion Matrix Heatmap ────────────────────────────────────
    print("\n[13] Saving Confusion Matrix heatmap...")
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        plt.title('XGBoost Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("    Saved: confusion_matrix.png")
    except Exception as plot_err:
        print(f"    Warning: Skipping Step 13 heatmap due to: {plot_err}")

    # ─── STEP 14: Model Comparison Bar Chart ──────────────────────────────────
    print("\n[14] Saving Model Comparison chart...")
    try:
        model_names = ['Decision\nTree', 'Random\nForest', 'KNN', 'Gradient\nBoosting', 'XGBoost\n(Tuned)']
        train_accs  = [results[k]['train'] * 100 for k in ['dt', 'rf', 'knn', 'gb', 'xgb']]
        test_accs   = [results[k]['test']  * 100 for k in ['dt', 'rf', 'knn', 'gb', 'xgb']]

        x = np.arange(len(model_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, train_accs, width, label='Training Accuracy',
                       color='#8a2be2', alpha=0.85)
        bars2 = ax.bar(x + width/2, test_accs,  width, label='Testing Accuracy',
                       color='#00e5ff', alpha=0.85)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison — All Classifiers')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylim(60, 100)
        ax.legend()
        for bar in list(bars1) + list(bars2):
            ax.annotate(f'{bar.get_height():.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/model_comparison.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("    Saved: model_comparison.png")
    except Exception as plot_err:
        print(f"    Warning: Skipping Step 14 comparison due to: {plot_err}")

    # ─── STEP 15: Save All Artifacts ──────────────────────────────────────────
    print("\n[15] Saving model artifacts to disk...")

    model_bundle = {
        'model':    xgb,
        'scaler':   sc,
        'features': features,
        'metrics': {
            'dt':  {'train': results['dt']['train'],  'test': results['dt']['test']},
            'rf':  {'train': results['rf']['train'],  'test': results['rf']['test']},
            'knn': {'train': results['knn']['train'], 'test': results['knn']['test']},
            'gb':  {'train': results['gb']['train'],  'test': results['gb']['test']},
            'xgb': {'train': xgb_train_acc,           'test': xgb_test_acc},
        }
    }

    with open(os.path.join(script_dir, 'rdf.pkl'), 'wb') as f:
        pickle.dump(model_bundle, f)
    with open(os.path.join(script_dir, 'scale1.pkl'), 'wb') as f:
        pickle.dump(sc, f)
    with open(os.path.join(script_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(model_bundle['metrics'], f)

    print("    Saved: rdf.pkl  (full bundle: model + scaler + features + metrics)")
    print("    Saved: scale1.pkl  (scaler only)")
    print("    Saved: metrics.pkl (metrics only)")

    print("\n" + "=" * 60)
    print("   Pipeline Complete!")
    print(f"   XGBoost -> Train: {xgb_train_acc*100:.2f}%  |  Test: {xgb_test_acc*100:.2f}%")
    print("=" * 60)



if __name__ == '__main__':
    main()
