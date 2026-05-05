"""
prediction_helper.py — Multi-Model Prediction Engine with SHAP Explainability

Supports Logistic Regression, XGBoost, LightGBM, and Random Forest models.
Provides SHAP-based feature explanations for each prediction.

Research Papers Referenced:
  - Paper 1 (IEEE 10638034): Multi-model ensemble comparison
  - Paper 2 (IEEE 9050779):  SHAP explainability for transparency
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

# ─── Model Registry ──────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')

MODEL_NAMES = {
    'Logistic Regression': 'logistic_regression',
    'XGBoost': 'xgboost',
    'LightGBM': 'lightgbm',
    'Random Forest': 'random_forest'
}

# Cache for loaded models and explainers
_model_cache = {}
_explainer_cache = {}
_scaler_data = None
_feature_list = None


def _load_scaler_data():
    """Load scaler and feature list (cached)."""
    global _scaler_data, _feature_list

    if _scaler_data is None:
        scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler_data.joblib')
        if os.path.exists(scaler_path):
            _scaler_data = joblib.load(scaler_path)
        else:
            # Fallback to legacy model_data.joblib
            model_data = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_data.joblib'))
            _scaler_data = {
                'scaler': model_data['scaler'],
                'cols_to_scale': model_data['cols_to_scale']
            }

    if _feature_list is None:
        feature_path = os.path.join(ARTIFACTS_DIR, 'feature_list.joblib')
        if os.path.exists(feature_path):
            _feature_list = joblib.load(feature_path)
        else:
            model_data = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_data.joblib'))
            _feature_list = list(model_data['features'])

    return _scaler_data, _feature_list


def load_model(model_display_name):
    """Load a model by display name, with caching."""
    model_key = MODEL_NAMES.get(model_display_name, 'logistic_regression')

    if model_key not in _model_cache:
        model_path = os.path.join(ARTIFACTS_DIR, f'{model_key}_model.joblib')
        if os.path.exists(model_path):
            _model_cache[model_key] = joblib.load(model_path)
        else:
            # Fallback to legacy model_data.joblib for LR
            model_data = joblib.load(os.path.join(ARTIFACTS_DIR, 'model_data.joblib'))
            _model_cache[model_key] = model_data['model']

    return _model_cache[model_key]


def get_available_models():
    """Return list of available model display names."""
    available = []
    for display_name, key in MODEL_NAMES.items():
        model_path = os.path.join(ARTIFACTS_DIR, f'{key}_model.joblib')
        legacy_path = os.path.join(ARTIFACTS_DIR, 'model_data.joblib')
        if os.path.exists(model_path) or (key == 'logistic_regression' and os.path.exists(legacy_path)):
            available.append(display_name)
    return available if available else ['Logistic Regression']


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                  residence_type, loan_purpose, loan_type):
    """Prepare a single input row for prediction."""
    scaler_data, features = _load_scaler_data()
    scaler = scaler_data['scaler']
    cols_to_scale = scaler_data['cols_to_scale']

    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        # Dummy values for scaler compatibility
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])

    # Scale the columns that the scaler expects
    available_scale_cols = [c for c in cols_to_scale if c in df.columns]
    df[available_scale_cols] = scaler.transform(df[available_scale_cols])

    # Select only the features used by the model
    df = df[features]
    return df


def calculate_credit_score(default_probability, base_score=300, scale_length=600):
    """Convert default probability to credit score (300-900) and rating."""
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability * scale_length
    credit_score = int(np.clip(credit_score, 300, 900))

    if 300 <= credit_score < 500:
        rating = 'Poor'
    elif 500 <= credit_score < 650:
        rating = 'Average'
    elif 650 <= credit_score < 750:
        rating = 'Good'
    elif 750 <= credit_score <= 900:
        rating = 'Excellent'
    else:
        rating = 'Undefined'

    return credit_score, rating


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type, model_name='Logistic Regression'):
    """
    Run prediction with the selected model.
    Returns: (default_probability, credit_score, rating)
    """
    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    model = load_model(model_name)

    # Get probability
    if hasattr(model, 'predict_proba'):
        default_probability = model.predict_proba(input_df)[:, 1][0]
    else:
        # Fallback for models without predict_proba
        x = np.dot(input_df.values, model.coef_.T) + model.intercept_
        default_probability = float(1 / (1 + np.exp(-x)))

    credit_score, rating = calculate_credit_score(default_probability)
    return default_probability, credit_score, rating


def get_shap_explanation(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                         delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                         residence_type, loan_purpose, loan_type, model_name='Logistic Regression'):
    """
    Get SHAP explanation for a single prediction.
    Returns: dict with 'shap_values', 'base_value', 'feature_names', 'feature_values'

    Research Paper 2 (IEEE 9050779): SHAP for model transparency.
    """
    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    model_key = MODEL_NAMES.get(model_name, 'logistic_regression')

    # Try loading pre-computed explainer
    if model_key not in _explainer_cache:
        explainer_path = os.path.join(ARTIFACTS_DIR, f'shap_{model_key}.joblib')
        if os.path.exists(explainer_path):
            _explainer_cache[model_key] = joblib.load(explainer_path)
        else:
            # Create explainer on-the-fly
            model = load_model(model_name)
            if model_key == 'logistic_regression':
                _explainer_cache[model_key] = shap.LinearExplainer(model, input_df)
            else:
                _explainer_cache[model_key] = shap.TreeExplainer(model)

    explainer = _explainer_cache[model_key]
    shap_values = explainer.shap_values(input_df)

    # Handle different SHAP output shapes
    if isinstance(shap_values, list):
        # For multi-class, take the positive class
        shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif len(shap_values.shape) == 3:
        shap_vals = shap_values[0, :, 1]
    else:
        shap_vals = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    # Human-readable feature names
    feature_labels = {
        'age': 'Age',
        'loan_tenure_months': 'Loan Tenure',
        'number_of_open_accounts': 'Open Accounts',
        'credit_utilization_ratio': 'Credit Utilization',
        'loan_to_income': 'Loan-to-Income Ratio',
        'delinquency_ratio': 'Delinquency Ratio',
        'avg_dpd_per_delinquency': 'Avg Days Past Due',
        'residence_type_Owned': 'Owns Residence',
        'residence_type_Rented': 'Rents Residence',
        'loan_purpose_Education': 'Education Loan',
        'loan_purpose_Home': 'Home Loan',
        'loan_purpose_Personal': 'Personal Loan',
        'loan_type_Unsecured': 'Unsecured Loan'
    }

    _, features = _load_scaler_data()
    display_names = [feature_labels.get(f, f) for f in features]

    return {
        'shap_values': shap_vals.tolist() if hasattr(shap_vals, 'tolist') else list(shap_vals),
        'base_value': float(base_value),
        'feature_names': display_names,
        'feature_values': input_df.values[0].tolist()
    }


def load_metrics():
    """Load model performance metrics."""
    metrics_path = os.path.join(ARTIFACTS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def predict_batch(df_input, model_name='Logistic Regression'):
    """
    Run batch predictions on a DataFrame.
    Expected columns: age, income, loan_amount, loan_tenure_months,
                      avg_dpd_per_delinquency, delinquency_ratio,
                      credit_utilization_ratio, num_open_accounts,
                      residence_type, loan_purpose, loan_type

    Research Paper 5 (IEEE 9002821): Batch processing for production systems.
    """
    results = []
    for _, row in df_input.iterrows():
        try:
            prob, score, rating = predict(
                age=int(row.get('age', 30)),
                income=float(row.get('income', 1000000)),
                loan_amount=float(row.get('loan_amount', 500000)),
                loan_tenure_months=int(row.get('loan_tenure_months', 36)),
                avg_dpd_per_delinquency=float(row.get('avg_dpd_per_delinquency', 0)),
                delinquency_ratio=float(row.get('delinquency_ratio', 0)),
                credit_utilization_ratio=float(row.get('credit_utilization_ratio', 30)),
                num_open_accounts=int(row.get('num_open_accounts', 2)),
                residence_type=str(row.get('residence_type', 'Owned')),
                loan_purpose=str(row.get('loan_purpose', 'Personal')),
                loan_type=str(row.get('loan_type', 'Unsecured')),
                model_name=model_name
            )
            paths = generate_improvement_paths(
                age=int(row.get('age', 30)),
                income=float(row.get('income', 1000000)),
                loan_amount=float(row.get('loan_amount', 500000)),
                loan_tenure_months=int(row.get('loan_tenure_months', 36)),
                avg_dpd_per_delinquency=float(row.get('avg_dpd_per_delinquency', 0)),
                delinquency_ratio=float(row.get('delinquency_ratio', 0)),
                credit_utilization_ratio=float(row.get('credit_utilization_ratio', 30)),
                num_open_accounts=int(row.get('num_open_accounts', 2)),
                residence_type=str(row.get('residence_type', 'Owned')),
                loan_purpose=str(row.get('loan_purpose', 'Personal')),
                loan_type=str(row.get('loan_type', 'Unsecured')),
                credit_score=score,
                default_probability=prob
            )
            
            if paths:
                top_path = paths[0]
                plan = top_path['inflation_note'] + " || Actions: " + "; ".join(top_path['actions'])
            else:
                plan = "Maintain good credit."

            results.append({
                'default_probability': round(prob, 4),
                'credit_score': score,
                'rating': rating,
                'improvement_plan': plan
            })
        except Exception as e:
            results.append({
                'default_probability': None,
                'credit_score': None,
                'rating': f'Error: {str(e)}',
                'improvement_plan': None
            })

    results_df = pd.DataFrame(results)
    return pd.concat([df_input.reset_index(drop=True), results_df], axis=1)


def generate_improvement_paths(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                               delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                               residence_type, loan_purpose, loan_type, credit_score, default_probability):
    """
    Generate 3 improvement paths with timelines to improve credit score and loan approval chances.
    Returns: list of paths with descriptions, timelines, and expected outcomes.
    
    Research Paper 6: Actionable Recourse & Counterfactual Explanations in Credit Risk (IEEE)
    """
    paths = []
    
    # Calculate current metrics
    current_score = credit_score
    target_score = 650  # Minimum for "Good" rating
    
    # Path 1: Debt Reduction Path
    if credit_utilization_ratio > 30 or delinquency_ratio > 20:
        actions = []
        timeline_months = 0
        
        if credit_utilization_ratio > 30:
            target_util = max(credit_utilization_ratio - 20, 10)
            actions.append(f"💳 Credit Card Usage: You are currently using {credit_utilization_ratio}% of your available credit limits. Try to bring this down to {target_util}% by paying off your credit card balances. Keeping your balances low shows lenders you manage money responsibly.")
            timeline_months = max(timeline_months, 6)
        
        if delinquency_ratio > 20:
            target_delinq = max(delinquency_ratio - 15, 0)
            actions.append(f"⚠️ Late Payments: Your current late payment rate is {delinquency_ratio}%. Aim to lower this to {target_delinq}% by paying any past-due bills as soon as possible. Paying on time is the single biggest factor in your credit score.")
            timeline_months = max(timeline_months, 12)
            
        actions.append("📅 Never Miss a Payment: Set up automatic payments with your bank so you never miss a due date again. A perfect payment history builds trust quickly.")
        
        if actions:
            projected_score = min(current_score + 80, 850)
            paths.append({
                'name': 'Debt Reduction Path',
                'description': 'Targeted reduction of existing debt metrics to quickly improve score',
                'actions': actions,
                'timeline_months': timeline_months,
                'projected_score': projected_score,
                'approval_likelihood': 'High' if projected_score >= 650 else 'Medium',
                'priority': 1
            })
    
    # Path 2: Income & Stability Path
    actions = []
    timeline_months = 0
    current_lti = loan_to_income_ratio(loan_amount, income)
    
    if current_lti > 2:
        target_loan = income * 1.5
        actions.append(f"⚖️ Loan Size vs. Income: Right now, you are asking for a loan that is {current_lti:.2f} times your annual income. Lenders prefer this to be under 2.0x. Consider asking for a smaller loan (around ₹{target_loan:,.0f}) or adding a co-applicant to share the load.")
        timeline_months = 6
    
    if residence_type == 'Rented':
        actions.append("🏠 Housing Stability: You currently rent your home. Lenders often see homeowners as more stable. If possible, consider adding a family member who owns a home as a co-signer to strengthen your application.")
        timeline_months = max(timeline_months, 12)
    
    if num_open_accounts < 2:
        actions.append(f"🏦 Credit Mix: You currently have {num_open_accounts} open credit accounts. Lenders like to see 2 or 3 active accounts. Consider safely opening a new basic credit card and paying it off completely every month to build your history.")
        timeline_months = max(timeline_months, 12)
    
    if actions:
        projected_score = min(current_score + 50, 800)
        paths.append({
            'name': 'Income & Stability Path',
            'description': 'Counterfactual path focusing on structural financial profile changes',
            'actions': actions,
            'timeline_months': timeline_months if timeline_months > 0 else 6,
            'projected_score': projected_score,
            'approval_likelihood': 'Medium' if projected_score >= 600 else 'Low',
            'priority': 2
        })
    
    # Path 3: Credit Building Track (Always present)
    actions = []
    timeline_months = 0
    
    if avg_dpd_per_delinquency > 30:
        actions.append(f"⏳ Overdue Time: On average, your late payments are {avg_dpd_per_delinquency} days overdue. Try to keep this under 15 days. If you are struggling, reach out to your lender to discuss a payment plan before things get worse.")
        timeline_months = 9
    
    if loan_type == 'Unsecured':
        actions.append("🛡️ Provide Security: You applied for an 'Unsecured' loan, which is riskier for the bank. Offering collateral (like a Fixed Deposit or a vehicle) can significantly improve your chances of getting approved.")
        timeline_months = max(timeline_months, 3)
        
    if num_open_accounts >= 2:
        actions.append("🕰️ Keep Old Accounts Open: The older your credit accounts are, the better. Don't close your oldest credit cards, even if you don't use them much, because they show you have a long, reliable history.")
        timeline_months = max(timeline_months, 12)
    else:
        actions.append("🌱 Build Your History: Start small. Get a basic credit card, use it for small everyday purchases, and pay the entire bill off every single month. Within 6 to 12 months, your score will steadily grow.")
        timeline_months = 12
    
    actions.append("🔍 Check Your Report: Get a free copy of your credit report once a month and make sure there are no errors or fraudulent accounts dragging your score down.")
    
    projected_score = min(current_score + 100, 900)
    paths.append({
        'name': 'Long-term Credit Building Track',
        'description': 'A systemic tracking mechanism for continuous credit score improvement',
        'actions': actions,
        'timeline_months': timeline_months if timeline_months > 0 else 12,
        'projected_score': projected_score,
        'approval_likelihood': 'High' if projected_score >= 700 else 'Medium',
        'priority': 3
    })
    
    # Sort by priority
    paths.sort(key=lambda x: x['priority'])
    
    # Add approval timeline for each path
    for path in paths:
        if path['projected_score'] >= 750:
            path['approval_months'] = min(path['timeline_months'], 6)
        elif path['projected_score'] >= 650:
            path['approval_months'] = min(path['timeline_months'] + 3, 12)
        elif path['projected_score'] >= 550:
            path['approval_months'] = min(path['timeline_months'] + 6, 18)
        else:
            path['approval_months'] = min(path['timeline_months'] + 12, 24)
            
        path['inflation_note'] = f"💡 Inflation Tracker: Central bank rates are projected to rise. Hitting the {path['projected_score']} target in {path['timeline_months']} months could save you up to 1.5% in interest against inflation."
    
    return paths


def loan_to_income_ratio(loan_amount, income):
    """Calculate loan-to-income ratio."""
    if income <= 0:
        return 0
    return loan_amount / income


def generate_survival_curve(default_probability, tenure_months):
    """
    Generate a synthetic survival curve (probability of not defaulting over time)
    based on a Weibull distribution where risk peaks around 1/3 of the tenure.
    
    Research Paper: Survival Analysis for Predicting Time to Default.
    """
    import numpy as np
    import pandas as pd
    
    # X-axis: months from 1 to tenure
    months = np.arange(1, tenure_months + 1)
    
    # Weibull cumulative distribution function shape: F(t) = 1 - exp(-(t/lambda)^k)
    k = 2.0
    
    # Prevent math error if default_prob is exactly 0
    safe_prob = max(min(default_probability, 0.999), 0.001)
    
    lam = tenure_months / ((-np.log(1 - safe_prob)) ** (1/k))
    
    # Survival function S(t) = exp(-(t/lam)^k)
    survival_probs = np.exp(-((months / lam) ** k))
    
    return pd.DataFrame({
        'Month': months,
        'Survival Probability': survival_probs
    })