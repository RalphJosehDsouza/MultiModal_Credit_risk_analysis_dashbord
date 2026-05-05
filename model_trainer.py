"""
model_trainer.py — Multi-Model Training Pipeline for Credit Risk Prediction

Trains 4 models (Logistic Regression, XGBoost, LightGBM, Random Forest) using
SMOTE-Tomek for class imbalance and Optuna for hyperparameter tuning.

Research Papers Referenced:
  - Paper 1 (IEEE 10638034): Ensemble Classifiers + SMOTE-ENN + SHAP
  - Paper 3 (IEEE 10914916): Ensemble Models + XAI for Regulatory Compliance
  - Paper 4 (IEEE 8845444):  ML Methods Comparison for Credit Risk

Usage:
  python model_trainer.py              # Generate synthetic data + train all models
  python model_trainer.py --data-dir . # Use real CSVs (customers.csv, loans.csv, bureau_data.csv)
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
import shap

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Feature Configuration (matches the original model) ──────────────────────
FEATURES_USED = [
    'age', 'loan_tenure_months', 'number_of_open_accounts',
    'credit_utilization_ratio', 'loan_to_income', 'delinquency_ratio',
    'avg_dpd_per_delinquency', 'residence_type_Owned',
    'residence_type_Rented', 'loan_purpose_Education', 'loan_purpose_Home',
    'loan_purpose_Personal', 'loan_type_Unsecured'
]

COLS_TO_SCALE = [
    'age', 'number_of_dependants', 'years_at_current_address', 'zipcode',
    'sanction_amount', 'processing_fee', 'gst', 'net_disbursement',
    'loan_tenure_months', 'principal_outstanding',
    'bank_balance_at_application', 'number_of_open_accounts',
    'number_of_closed_accounts', 'enquiry_count',
    'credit_utilization_ratio', 'loan_to_income', 'delinquency_ratio',
    'avg_dpd_per_delinquency'
]


def generate_synthetic_data(n_samples=50000, default_rate=0.10, random_state=42):
    """
    Generate synthetic credit data matching the original dataset distributions.
    Uses the known min/max from the original scaler to produce realistic ranges.
    """
    rng = np.random.RandomState(random_state)

    # Demographics
    age = rng.randint(18, 71, n_samples)
    income = rng.randint(300000, 6000000, n_samples)
    number_of_dependants = rng.randint(0, 6, n_samples)
    years_at_current_address = rng.randint(1, 32, n_samples)
    zipcode = rng.choice([110001, 400001, 560001, 600001, 700001, 411001], n_samples)

    residence_type = rng.choice(['Owned', 'Rented', 'Mortgage'], n_samples, p=[0.5, 0.35, 0.15])
    loan_purpose = rng.choice(['Education', 'Home', 'Auto', 'Personal'], n_samples, p=[0.15, 0.3, 0.25, 0.3])
    loan_type = rng.choice(['Secured', 'Unsecured'], n_samples, p=[0.55, 0.45])

    # Loan details
    loan_amount = rng.randint(100000, 5500000, n_samples)
    sanction_amount = (loan_amount * rng.uniform(1.05, 1.3, n_samples)).astype(int)
    processing_fee = (loan_amount * rng.uniform(0.01, 0.025, n_samples)).astype(int)
    gst = (processing_fee * 18).astype(int)
    net_disbursement = (loan_amount * rng.uniform(0.75, 0.95, n_samples)).astype(int)
    loan_tenure_months = rng.choice([6, 12, 18, 24, 30, 36, 48, 59], n_samples)
    principal_outstanding = (loan_amount * rng.uniform(0.3, 0.95, n_samples)).astype(int)
    bank_balance_at_application = rng.randint(19415, 7846644, n_samples)

    # Bureau data
    number_of_open_accounts = rng.randint(1, 5, n_samples)
    number_of_closed_accounts = rng.randint(0, 3, n_samples)
    total_loan_months = rng.randint(6, 180, n_samples)
    delinquent_months = rng.randint(0, 40, n_samples)
    delinquent_months = np.minimum(delinquent_months, total_loan_months)
    total_dpd = delinquent_months * rng.uniform(0, 12, n_samples)
    enquiry_count = rng.randint(1, 10, n_samples)
    credit_utilization_ratio = rng.randint(0, 100, n_samples)

    # Derived features
    loan_to_income = np.round(loan_amount / income, 2)
    delinquency_ratio = np.round((delinquent_months * 100 / total_loan_months), 1)
    avg_dpd_per_delinquency = np.where(
        delinquent_months > 0,
        np.round(total_dpd / delinquent_months, 1),
        0
    )

    # Generate default label (correlated with risk features)
    risk_score = (
        0.3 * (delinquency_ratio / 100) +
        0.25 * (credit_utilization_ratio / 100) +
        0.15 * (loan_to_income / 5) +
        0.1 * (avg_dpd_per_delinquency / 10) +
        0.1 * (1 - age / 70) +
        0.05 * (loan_tenure_months / 59) +
        0.05 * np.where(loan_type == 'Unsecured', 1, 0)
    )
    risk_score = np.clip(risk_score, 0, 1)

    # Calibrate threshold to get ~10% default rate
    threshold = np.percentile(risk_score, 100 * (1 - default_rate))
    noise = rng.uniform(-0.05, 0.05, n_samples)
    default = ((risk_score + noise) > threshold).astype(int)

    # One-hot encode categorical features
    df = pd.DataFrame({
        'age': age,
        'number_of_dependants': number_of_dependants,
        'years_at_current_address': years_at_current_address,
        'zipcode': zipcode,
        'sanction_amount': sanction_amount,
        'processing_fee': processing_fee,
        'gst': gst,
        'net_disbursement': net_disbursement,
        'loan_tenure_months': loan_tenure_months,
        'principal_outstanding': principal_outstanding,
        'bank_balance_at_application': bank_balance_at_application,
        'number_of_open_accounts': number_of_open_accounts,
        'number_of_closed_accounts': number_of_closed_accounts,
        'enquiry_count': enquiry_count,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_to_income,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': (residence_type == 'Owned').astype(int),
        'residence_type_Rented': (residence_type == 'Rented').astype(int),
        'loan_purpose_Education': (loan_purpose == 'Education').astype(int),
        'loan_purpose_Home': (loan_purpose == 'Home').astype(int),
        'loan_purpose_Personal': (loan_purpose == 'Personal').astype(int),
        'loan_type_Unsecured': (loan_type == 'Unsecured').astype(int),
        'default': default
    })

    print(f"  Generated {n_samples} samples | Default rate: {default.mean():.1%}")
    return df


def load_real_data(data_dir):
    """Load and preprocess the real CSV data (same pipeline as the notebook)."""
    df_customers = pd.read_csv(f"{data_dir}/customers.csv")
    df_loans = pd.read_csv(f"{data_dir}/loans.csv")
    df_bureau = pd.read_csv(f"{data_dir}/bureau_data.csv")

    df = pd.merge(df_customers, df_loans, on='cust_id')
    df = pd.merge(df, df_bureau, on='cust_id')

    # Feature engineering
    df['loan_to_income'] = round(df['loan_amount'] / df['income'], 2)
    df['delinquency_ratio'] = round(df['delinquent_months'] * 100 / df['total_loan_months'], 1)
    df['avg_dpd_per_delinquency'] = np.where(
        df['delinquent_months'] > 0,
        round(df['total_dpd'] / df['delinquent_months'], 1),
        0
    )

    # Drop unused columns
    cols_to_drop = ['cust_id', 'loan_id', 'disbursal_date', 'installment_start_dt',
                    'loan_amount', 'income', 'total_loan_months', 'delinquent_months',
                    'total_dpd', 'gender', 'marital_status', 'employment_status',
                    'city', 'state']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis='columns')

    # One-hot encode
    cat_cols = ['residence_type', 'loan_purpose', 'loan_type']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Ensure boolean columns are int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print(f"  Loaded {len(df)} samples from CSV | Default rate: {df['default'].mean():.1%}")
    return df


def train_models(df):
    """Train 4 models with SMOTE-Tomek + Optuna, return models and metrics."""

    # ── Split ────────────────────────────────────────────────────────────────
    X = df.drop('default', axis='columns')
    y = df['default'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    # ── Scale ────────────────────────────────────────────────────────────────
    available_cols_to_scale = [c for c in COLS_TO_SCALE if c in X_train.columns]
    scaler = MinMaxScaler()
    X_train[available_cols_to_scale] = scaler.fit_transform(X_train[available_cols_to_scale])
    X_test[available_cols_to_scale] = scaler.transform(X_test[available_cols_to_scale])

    # Select final features
    available_features = [f for f in FEATURES_USED if f in X_train.columns]
    X_train_feat = X_train[available_features]
    X_test_feat = X_test[available_features]

    # ── SMOTE-Tomek ──────────────────────────────────────────────────────────
    print("  Applying SMOTE-Tomek...")
    smt = SMOTETomek(random_state=42)
    X_train_smt, y_train_smt = smt.fit_resample(X_train_feat, y_train)
    print(f"  After SMOTE-Tomek: {len(X_train_smt)} samples | Balance: {y_train_smt.value_counts().to_dict()}")

    # ── Define model configs ─────────────────────────────────────────────────
    models = {}
    metrics = {}

    # 1) Logistic Regression with Optuna
    print("\n  [1/4] Training Logistic Regression with Optuna...")

    def lr_objective(trial):
        param = {
            'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg']),
            'tol': trial.suggest_float('tol', 1e-6, 1e-1, log=True),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        model = LogisticRegression(**param, max_iter=10000)
        scores = cross_val_score(model, X_train_smt, y_train_smt, cv=3, scoring='f1_macro', n_jobs=-1)
        return np.mean(scores)

    study_lr = optuna.create_study(direction='maximize')
    study_lr.optimize(lr_objective, n_trials=30)
    lr_model = LogisticRegression(**study_lr.best_params, max_iter=10000)
    lr_model.fit(X_train_smt, y_train_smt)
    models['logistic_regression'] = lr_model
    print(f"    Best F1: {study_lr.best_value:.4f}")

    # 2) XGBoost with Optuna
    print("  [2/4] Training XGBoost with Optuna...")

    def xgb_objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        model = XGBClassifier(**param, random_state=42)
        scores = cross_val_score(model, X_train_smt, y_train_smt, cv=3, scoring='f1_macro', n_jobs=-1)
        return np.mean(scores)

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=30)
    xgb_model = XGBClassifier(**study_xgb.best_params, eval_metric='logloss', verbosity=0, random_state=42)
    xgb_model.fit(X_train_smt, y_train_smt)
    models['xgboost'] = xgb_model
    print(f"    Best F1: {study_xgb.best_value:.4f}")

    # 3) LightGBM with Optuna
    print("  [3/4] Training LightGBM with Optuna...")

    def lgbm_objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'verbose': -1
        }
        model = LGBMClassifier(**param, random_state=42)
        scores = cross_val_score(model, X_train_smt, y_train_smt, cv=3, scoring='f1_macro', n_jobs=-1)
        return np.mean(scores)

    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lgbm_objective, n_trials=30)
    lgbm_model = LGBMClassifier(**study_lgbm.best_params, verbose=-1, random_state=42)
    lgbm_model.fit(X_train_smt, y_train_smt)
    models['lightgbm'] = lgbm_model
    print(f"    Best F1: {study_lgbm.best_value:.4f}")

    # 4) Random Forest with Optuna
    print("  [4/4] Training Random Forest with Optuna...")

    def rf_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        model = RandomForestClassifier(**param, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_train_smt, y_train_smt, cv=3, scoring='f1_macro', n_jobs=-1)
        return np.mean(scores)

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(rf_objective, n_trials=30)
    rf_model = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_smt, y_train_smt)
    models['random_forest'] = rf_model
    print(f"    Best F1: {study_rf.best_value:.4f}")

    # ── Evaluate all models ──────────────────────────────────────────────────
    print("\n  Evaluating models on test set...")
    for name, model in models.items():
        y_pred = model.predict(X_test_feat)
        y_proba = model.predict_proba(X_test_feat)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        metrics[name] = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'auc_roc': round(roc_auc_score(y_test, y_proba), 4),
            'confusion_matrix': cm.tolist(),
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'pr_curve': {'precision': prec_curve.tolist(), 'recall': rec_curve.tolist()}
        }
        print(f"    {name:25s} | Acc: {metrics[name]['accuracy']:.4f} | "
              f"F1: {metrics[name]['f1']:.4f} | AUC: {metrics[name]['auc_roc']:.4f}")

    return models, metrics, scaler, available_cols_to_scale, available_features, X_train_smt, X_test_feat, y_test


def save_artifacts(models, metrics, scaler, cols_to_scale, features, X_train_smt):
    """Save all model artifacts to the artifacts directory."""
    import os
    os.makedirs('artifacts', exist_ok=True)

    # Save individual models
    for name, model in models.items():
        joblib.dump(model, f'artifacts/{name}_model.joblib')
        print(f"  Saved artifacts/{name}_model.joblib")

    # Save backward-compatible model_data.joblib (LR model)
    model_data = {
        'model': models['logistic_regression'],
        'features': pd.Index(features),
        'scaler': scaler,
        'cols_to_scale': pd.Index(cols_to_scale)
    }
    joblib.dump(model_data, 'artifacts/model_data.joblib')
    print("  Saved artifacts/model_data.joblib (backward compatible)")

    # Save metrics
    with open('artifacts/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  Saved artifacts/model_metrics.json")

    # Save SHAP explainers
    print("  Computing SHAP explainers (this may take a minute)...")
    X_sample = X_train_smt.sample(min(500, len(X_train_smt)), random_state=42)
    for name, model in models.items():
        try:
            if name == 'logistic_regression':
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                explainer = shap.TreeExplainer(model)
            joblib.dump(explainer, f'artifacts/shap_{name}.joblib')
            print(f"  Saved artifacts/shap_{name}.joblib")
        except Exception as e:
            print(f"  Warning: Could not save SHAP explainer for {name}: {e}")

    # Save feature list
    joblib.dump(features, 'artifacts/feature_list.joblib')
    print("  Saved artifacts/feature_list.joblib")

    # Save scaler separately for multi-model use
    joblib.dump({'scaler': scaler, 'cols_to_scale': cols_to_scale}, 'artifacts/scaler_data.joblib')
    print("  Saved artifacts/scaler_data.joblib")


def main():
    parser = argparse.ArgumentParser(description='Train credit risk models')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing customers.csv, loans.csv, bureau_data.csv')
    parser.add_argument('--samples', type=int, default=50000,
                        help='Number of synthetic samples to generate (if no data-dir)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Credit Risk Model Training Pipeline")
    print("  Papers: IEEE 10638034, 10914916, 8845444")
    print("=" * 60)

    # Load or generate data
    if args.data_dir:
        print(f"\n[DATA] Loading real data from {args.data_dir}...")
        df = load_real_data(args.data_dir)
    else:
        print(f"\n[GEN] Generating {args.samples} synthetic samples...")
        df = generate_synthetic_data(n_samples=args.samples)

    # Train models
    print("\n[TRAIN] Training 4 models with SMOTE-Tomek + Optuna...")
    models, metrics, scaler, cols_to_scale, features, X_train_smt, X_test, y_test = train_models(df)

    # Save artifacts
    print("\n[SAVE] Saving artifacts...")
    save_artifacts(models, metrics, scaler, cols_to_scale, features, X_train_smt)

    print("\n" + "=" * 60)
    print("  [DONE] Training complete! All artifacts saved to ./artifacts/")
    print("=" * 60)
    print("\nModel Performance Summary:")
    for name, m in metrics.items():
        print(f"  {name:25s} | Accuracy: {m['accuracy']:.2%} | AUC-ROC: {m['auc_roc']:.4f}")


if __name__ == '__main__':
    main()
