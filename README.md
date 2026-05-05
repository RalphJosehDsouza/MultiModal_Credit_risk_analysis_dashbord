# Credit Risk Intelligence Platform

## Overview

A comprehensive, end-to-end machine learning platform for credit risk assessment that predicts loan default probability, generates creditworthiness ratings, and provides transparent SHAP-based explanations for every prediction. Built with Python, Streamlit, and backed by 5 IEEE research papers.

The platform trains and compares **four classification models** — Logistic Regression, XGBoost, LightGBM, and Random Forest — on a dataset of ~50,000 loan applications. Class imbalance is handled via **SMOTE-Tomek** resampling, and hyperparameters are optimised using **Optuna** Bayesian search (30 trials per model).

**Key Metrics (Best Model — Logistic Regression):**
| Metric | Value |
|--------|-------|
| Accuracy | 99.63% |
| Precision | 99.63% |
| Recall | 99.63% |
| F1-Score | 99.63% |
| AUC-ROC | 0.9963 |

This tool is designed for banks, fintech platforms, credit unions, and lending institutions seeking to optimise loan approval processes and manage credit risk effectively.

---

## Key Features

### 🎯 Multi-Model Credit Risk Prediction
- **4 models trained and compared:** Logistic Regression, XGBoost, LightGBM, Random Forest
- Auto-selects the best-performing model based on AUC-ROC score
- Generates a **credit score (300–900)** and categorical rating (Poor / Average / Good / Excellent)
- Supports **individual prediction** (manual entry, preset profiles, single CSV) and **batch prediction** (CSV upload with per-borrower results)

### 🔍 SHAP Explainability Dashboard
- Per-prediction SHAP bar chart showing exactly which features drove the decision
- Red bars = features increasing default risk; Green bars = features decreasing risk
- Satisfies regulatory transparency requirements (GDPR, RBI Fair Lending Guidelines)

### 📊 Model Performance Analytics
- **ROC Curves with Shaded AUC Area** — full-width, interactive visualisation for all 4 models
- **AUC-ROC Comparison Bar Chart** — side-by-side model comparison with random baseline reference
- **Precision-Recall Curves** — model discrimination at various thresholds
- **Confusion Matrices** — per-model heatmaps with TP/TN/FP/FN counts
- **Metrics Comparison Table** — accuracy, precision, recall, F1, AUC for all models
- **Algorithmic Fairness Audit** — disparate impact analysis across age and residence demographics

### 🌊 Advanced Assessment Features
- **Macroeconomic Stress Testing** — simulate mild, moderate, and severe economic shocks to see how borrower risk changes under adverse conditions
- **Survival Analysis** — time-to-default probability curves showing when a borrower is most likely to default
- **Personalised Improvement Paths** — actionable step-by-step credit improvement plans for low-scoring borrowers with projected score targets and approval timelines

### ⚙️ Smart Feature Engineering
- Auto-calculated derived features: Loan-to-Income Ratio, Delinquency Ratio, Average DPD per Delinquency
- Preset borrower profiles for quick testing (Young Professional, Senior Borrower, High Risk, etc.)
- Real-time input validation with warnings for out-of-range values

---

## Dataset

> **Source:** [Kaggle — Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  
> **Origin:** Tunisian Commercial Bank customer records  
> **Records:** ~50,000 loan applications across 3 relational tables  
> **Tables:** `customers.csv` (demographics), `loans.csv` (loan details + target), `bureau_data.csv` (credit history)  
> **Raw Features:** 24 variables → **14 engineered features** used for model training  
> **Target Variable:** `default` (binary — 0 = non-default, 1 = default)  
> **Class Distribution:** 90% non-default / 10% default (addressed via SMOTE-Tomek)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER (Streamlit)                         │
│  ┌──────────┐ ┌────────────────┐ ┌──────────────────┐   │
│  │   Risk   │ │     Model      │ │    About &       │   │
│  │Assessment│ │  Performance   │ │    Papers        │   │
│  └──────────┘ └────────────────┘ └──────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  APPLICATION LOGIC LAYER                                │
│  ┌──────────────┐ ┌──────────┐ ┌────────────────────┐   │
│  │ prediction   │ │  SHAP    │ │  Stress Test /     │   │
│  │ _helper.py   │ │ Explainer│ │  Fairness Audit    │   │
│  └──────────────┘ └──────────┘ └────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  MODEL LAYER                                            │
│  ┌─────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐      │
│  │ LR  │ │ XGBoost │ │ LightGBM│ │ Random Forest│      │
│  └─────┘ └─────────┘ └─────────┘ └──────────────┘      │
├─────────────────────────────────────────────────────────┤
│  DATA / ARTIFACT LAYER                                  │
│  4 Models (.joblib) │ 4 SHAP Explainers │ Scaler │ JSON │
└─────────────────────────────────────────────────────────┘
```

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **99.63%** | **99.63%** | **99.63%** | **99.63%** | **0.9963** |
| LightGBM | 99.41% | 99.41% | 99.41% | 99.41% | 0.9941 |
| XGBoost | 99.38% | 99.38% | 99.38% | 99.38% | 0.9938 |
| Random Forest | 99.10% | 99.10% | 99.10% | 99.10% | 0.9910 |

Logistic Regression is auto-selected as the best model due to the highest AUC-ROC (0.9963) and superior interpretability for regulatory compliance. All models achieve AUC-ROC > 0.99.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RalphJosehDsouza/Mentor_miniProject.git
   cd credit-risk-prediction-model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Required: pandas, numpy, scikit-learn, xgboost, lightgbm, imbalanced-learn, optuna, streamlit, plotly, shap, joblib

3. **Verify artifacts exist:**
   ```
   artifacts/
   ├── logistic_regression_model.joblib
   ├── xgboost_model.joblib
   ├── lightgbm_model.joblib
   ├── random_forest_model.joblib
   ├── logistic_regression_shap_explainer.joblib
   ├── xgboost_shap_explainer.joblib
   ├── lightgbm_shap_explainer.joblib
   ├── random_forest_shap_explainer.joblib
   ├── scaler_data.joblib
   └── model_metrics.json
   ```

4. **Launch the application:**
   ```bash
   streamlit run main.py
   ```
   Access at `http://localhost:8501`

---

## Usage Guide

### Input Parameters

**Borrower Demographics:**
- Age: 18–100 years
- Annual Income: Verified income in INR
- Residence Type: Owned, Rented, or Mortgage

**Loan Details:**
- Loan Amount: Requested principal amount (INR)
- Loan Tenure: Repayment period in months
- Loan Purpose: Education, Home, Auto, or Personal
- Loan Type: Secured or Unsecured

**Credit History:**
- Average DPD: Mean days past due from previous accounts
- Delinquency Ratio: Percentage of delinquent accounts
- Credit Utilization Ratio: Percentage of available credit used
- Number of Open Loan Accounts: Count of active credit accounts

### Output Interpretation

| Credit Score | Rating | Risk Level |
|-------------|--------|------------|
| 750–900 | Excellent | Low Risk |
| 650–749 | Good | Moderate Risk |
| 500–649 | Average | Elevated Risk |
| 300–499 | Poor | High Risk |

---

## Project Structure

```
credit-risk-prediction-model/
├── main.py                      # Streamlit dashboard (1100+ lines)
├── model_trainer.py             # Training pipeline with SMOTE-Tomek + Optuna
├── prediction_helper.py         # Inference engine with SHAP integration
├── requirements.txt             # Python dependencies
├── credit_risk_report.tex       # LaTeX project report
├── project_logbook.tex          # Weekly progress logbook
├── viva_prep_guide.md           # Viva preparation guide
├── artifacts/                   # Pre-trained models and metrics
│   ├── *_model.joblib           # 4 trained classification models
│   ├── *_shap_explainer.joblib  # 4 SHAP explainer objects
│   ├── scaler_data.joblib       # Fitted MinMaxScaler + feature list
│   └── model_metrics.json       # Performance metrics + ROC/PR curve data
└── data/                        # Source dataset
    ├── customers.csv            # Customer demographics
    ├── loans.csv                # Loan details + target variable
    └── bureau_data.csv          # Credit bureau history
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Class Imbalance | SMOTE-Tomek (imbalanced-learn) |
| Hyperparameter Tuning | Optuna (Bayesian Optimisation) |
| Frontend | Streamlit |
| Visualisation | Plotly |
| Data Processing | Pandas, NumPy |
| Serialisation | Joblib |
| Version Control | Git / GitHub |

---

## Research Papers

This project's features are each backed by a specific IEEE-published research paper:

| # | Paper | Feature Built |
|---|-------|---------------|
| 1 | [Effective Credit Risk Prediction Using Ensemble Classifiers With Model Explanation](https://ieeexplore.ieee.org/document/10638034) (IEEE Access, 2024) | Multi-model comparison pipeline (LR, XGBoost, LightGBM, RF) |
| 2 | [Explainability of ML Granting Scoring in P2P Lending](https://ieeexplore.ieee.org/document/9050779) (IEEE, 2020) | Interactive SHAP explainability dashboard |
| 3 | [Credit Risk Assessment using Ensemble Models and XAI](https://ieeexplore.ieee.org/document/10914916) (IEEE, 2024) | Performance analytics with ROC, AUC, confusion matrices |
| 4 | [Credit Risk Prediction Based on ML Methods](https://ieeexplore.ieee.org/document/8845444) (IEEE ICCSE, 2019) | Smart feature engineering & automated input system |
| 5 | [Incremental Learning Ensemble for Imbalanced Credit Scoring](https://ieeexplore.ieee.org/document/9002821) (IEEE, 2020) | Batch CSV prediction with improvement plans |

---

## Future Enhancements

- **Real-Time Data Integration:** Connect to live transaction feeds for dynamic risk assessment
- **REST API Development:** Expose model as FastAPI/Flask REST API for banking system integration
- **Model Monitoring:** Implement drift detection and automated retraining pipelines
- **Deep Learning:** Explore neural networks and transformer-based models for feature extraction
- **Cloud Deployment:** Deploy on AWS/GCP/Azure with auto-scaling
- **Regulatory Reporting:** Automated compliance report generation

---

## Team

| Name | Roll Number |
|------|-------------|
| Ralph Joseph Dsouza | 10242 |
| Chris Fernandes | 10244 |
| Ijas Keni | 10253 |

**Guide:** Prof. Sohan Agate  
**Principal:** Dr. Sapna Prabhu  
**Institution:** Fr. Conceicao Rodrigues College of Engineering, Bandra  
**University:** University of Mumbai  
**Academic Year:** 2025–2026