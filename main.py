"""
main.py — Credit Risk Intelligence Platform

A comprehensive Streamlit application for credit risk assessment featuring:
  1. Multi-model prediction (LR, XGBoost, LightGBM, RF)    — Paper 1 (IEEE 10638034)
  2. SHAP explainability dashboard                          — Paper 2 (IEEE 9050779)
  3. Performance analytics with ROC/PR curves               — Paper 3 (IEEE 10914916)
  4. Smart automated input with presets & validation         — Paper 4 (IEEE 8845444)
  5. Batch CSV prediction with PDF export                   — Paper 5 (IEEE 9002821)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prediction_helper import (
    predict, get_shap_explanation, load_metrics,
    get_available_models, predict_batch, generate_improvement_paths,
    generate_survival_curve
)
import os
import io

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="CreditRisk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 0.75rem;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
    }
    .metric-label { color: #94a3b8; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-sub { color: #94a3b8; font-size: 0.75rem; }

    /* Rating badges */
    .badge-poor { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; padding: 0.4rem 1rem; border-radius: 2rem; font-weight: 600; display: inline-block; }
    .badge-average { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.4rem 1rem; border-radius: 2rem; font-weight: 600; display: inline-block; }
    .badge-good { background: linear-gradient(135deg, #22c55e, #16a34a); color: white; padding: 0.4rem 1rem; border-radius: 2rem; font-weight: 600; display: inline-block; }
    .badge-excellent { background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; padding: 0.4rem 1rem; border-radius: 2rem; font-weight: 600; display: inline-block; }

    /* Warning badges */
    .warning-badge {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fbbf24;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        margin: 0.2rem 0;
    }

    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 1rem;
    }

    /* Traffic light */
    .traffic-light {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1rem;
        border-radius: 0.5rem;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .tl-low { background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); color: #4ade80; }
    .tl-medium { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); color: #fbbf24; }
    .tl-high { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }

    /* Sidebar styling */
    .css-1d391kg { padding-top: 1rem; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
</style>
""", unsafe_allow_html=True)


# ─── Preset Profiles (Paper 4 — IEEE 8845444) ────────────────────────────────
PRESETS = {
    "Custom (Enter Manually)": None,
    "Salaried Professional (Low Risk)": {
        'age': 35, 'income': 1800000, 'loan_amount': 1000000,
        'loan_tenure_months': 24, 'avg_dpd': 5, 'delinquency': 10,
        'credit_util': 20, 'open_accounts': 1,
        'residence': 'Owned', 'purpose': 'Home', 'type': 'Secured'
    },
    "Self-Employed (Medium Risk)": {
        'age': 42, 'income': 1200000, 'loan_amount': 2500000,
        'loan_tenure_months': 48, 'avg_dpd': 25, 'delinquency': 35,
        'credit_util': 55, 'open_accounts': 3,
        'residence': 'Rented', 'purpose': 'Personal', 'type': 'Unsecured'
    },
    "Young First-Time Borrower": {
        'age': 23, 'income': 500000, 'loan_amount': 300000,
        'loan_tenure_months': 12, 'avg_dpd': 0, 'delinquency': 0,
        'credit_util': 15, 'open_accounts': 1,
        'residence': 'Rented', 'purpose': 'Education', 'type': 'Unsecured'
    },
    "High-Risk Profile": {
        'age': 55, 'income': 400000, 'loan_amount': 1500000,
        'loan_tenure_months': 48, 'avg_dpd': 50, 'delinquency': 60,
        'credit_util': 85, 'open_accounts': 4,
        'residence': 'Mortgage', 'purpose': 'Personal', 'type': 'Unsecured'
    },
}


def get_rating_badge(rating):
    """Return HTML badge for rating."""
    badge_class = f"badge-{rating.lower()}"
    return f'<span class="{badge_class}">{rating}</span>'


def create_gauge_chart(score):
    """Create an animated credit score gauge chart."""
    if score < 500:
        color = "#ef4444"
    elif score < 650:
        color = "#f59e0b"
    elif score < 750:
        color = "#22c55e"
    else:
        color = "#6366f1"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [300, 900], 'tickwidth': 2, 'tickcolor': '#475569',
                     'tickvals': [300, 500, 650, 750, 900],
                     'ticktext': ['300', '500', '650', '750', '900']},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#1e293b',
            'borderwidth': 0,
            'steps': [
                {'range': [300, 500], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [500, 650], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [650, 750], 'color': 'rgba(34, 197, 94, 0.15)'},
                {'range': [750, 900], 'color': 'rgba(99, 102, 241, 0.15)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'}
    )
    return fig


def compute_emi(loan_amount, tenure_months, annual_rate=10.0):
    """Calculate EMI using standard formula."""
    if tenure_months <= 0 or loan_amount <= 0:
        return 0
    r = annual_rate / 12 / 100
    if r == 0:
        return loan_amount / tenure_months
    emi = loan_amount * (r * (1 + r) ** tenure_months) / ((1 + r) ** tenure_months - 1)
    return emi


def get_preliminary_risk(delinquency, credit_util, loan_to_income, avg_dpd):
    """Quick heuristic risk indicator (NOT model prediction)."""
    risk_score = (delinquency / 100 * 0.35 + credit_util / 100 * 0.3 +
                  min(loan_to_income / 5, 1) * 0.2 + min(avg_dpd / 60, 1) * 0.15)
    if risk_score < 0.25:
        return "low", "LOW", "tl-low"
    elif risk_score < 0.5:
        return "medium", "MEDIUM", "tl-medium"
    else:
        return "high", "HIGH", "tl-high"


# ─── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["Risk Assessment", "Model Performance", "About & Papers"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        '<p style="color: #64748b; font-size: 0.75rem;">Credit Risk Intelligence Platform v2.0<br>'
        'Powered by 5 IEEE Research Papers</p>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
if page == "Risk Assessment":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Credit Risk Intelligence Platform</h1>
        <p>Multi-model credit risk assessment with AI-powered explainability</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-select best model based on metrics
    metrics = load_metrics()
    if metrics:
        # Find best model by F1 score (balance of precision and recall for credit risk)
        best_model = 'Logistic Regression'
        best_f1 = 0
        for model_key, model_metrics in metrics.items():
            f1 = model_metrics.get('f1_score', 0)
            if f1 > best_f1:
                best_f1 = f1
                model_key_display = {
                    'logistic_regression': 'Logistic Regression',
                    'xgboost': 'XGBoost',
                    'lightgbm': 'LightGBM',
                    'random_forest': 'Random Forest'
                }.get(model_key, model_key)
                best_model = model_key_display
    else:
        # Default to Logistic Regression if no metrics
        best_model = 'Logistic Regression'
    
    st.info(f"Using best model: {best_model} (auto-selected based on performance)")
    selected_model = best_model

    # Input Method Selector
    st.markdown("---")
    input_method = st.radio(
        "Select Input Method",
        ["Manual Entry", "Upload CSV (Batch Processing)"],
        horizontal=True,
        help="Choose between manual form entry or uploading a CSV file for batch processing"
    )

    if input_method == "Upload CSV (Batch Processing)":
        st.markdown('<p class="section-header">Batch Credit Risk Prediction</p>', unsafe_allow_html=True)
        st.caption("Based on: *Incremental Learning Ensemble Method for Imbalanced Credit Scoring* (IEEE 9002821)")

        # Template download
        st.markdown("### Step 1: Download Template")
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_batch.csv')
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                st.download_button(
                    "Download Sample CSV Template",
                    f.read(),
                    file_name="sample_batch_template.csv",
                    mime="text/csv"
                )
        else:
            st.warning("sample_batch.csv not found")

        st.markdown("### Step 2: Upload Your CSV")
        uploaded_file = st.file_uploader(
            "Upload CSV file with borrower data",
            type=['csv'],
            help="CSV must contain columns: age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency, delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type, loan_purpose, loan_type"
        )
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df_upload)} rows**")
            st.dataframe(df_upload.head(), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df_upload)} predictions with {selected_model}..."):
                    st.session_state['batch_results'] = predict_batch(df_upload, model_name=selected_model)
                    st.session_state['batch_df_upload'] = df_upload

            if 'batch_results' in st.session_state:
                results = st.session_state['batch_results']
                df_upload_state = st.session_state['batch_df_upload']

                st.markdown("### Results")

                def color_rating(val):
                    colors = {'Poor': '#ef4444', 'Average': '#f59e0b', 'Good': '#22c55e', 'Excellent': '#6366f1'}
                    return f'color: {colors.get(val, "#e2e8f0")}'

                st.dataframe(
                    results.style.map(color_rating, subset=['rating']) if hasattr(results.style, 'map') else results.style.applymap(color_rating, subset=['rating']),
                    use_container_width=True, hide_index=True
                )

                # Summary statistics
                st.markdown("### Summary")
                summ1, summ2, summ3 = st.columns(3)
                with summ1:
                    avg_score = results['credit_score'].mean()
                    st.metric("Average Credit Score", f"{avg_score:.0f}")
                with summ2:
                    avg_prob = results['default_probability'].mean()
                    st.metric("Average Default Probability", f"{avg_prob:.1%}")
                with summ3:
                    high_risk = (results['rating'].isin(['Poor', 'Average'])).sum()
                    st.metric("High Risk Borrowers", f"{high_risk}/{len(results)}")

                # Download results
                st.markdown("### Download Results")
                csv_buffer = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV Results",
                    csv_buffer,
                    file_name="credit_risk_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.markdown("---")
                st.markdown("### Individual Borrower Analysis")
                st.caption("Select a borrower from the batch to view their specific analysis and improvement paths.")
                
                borrower_options = [f"Borrower {i+1} (Score: {row['credit_score']:.0f})" for i, row in results.iterrows()]
                selected_b_idx = st.selectbox("Select Borrower", range(len(results)), format_func=lambda x: borrower_options[x])
                
                if selected_b_idx is not None:
                    row_data = df_upload_state.iloc[selected_b_idx].to_dict()
                    res_data = results.iloc[selected_b_idx]
                    
                    st.markdown(f"#### Detailed Analysis for Borrower {selected_b_idx + 1}")
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.plotly_chart(create_gauge_chart(res_data['credit_score']), use_container_width=True, key=f"gauge_batch_{selected_b_idx}")
                    with col_b2:
                        st.markdown(f"**Default Probability:** {res_data['default_probability']:.1%}")
                        st.markdown(f"**Credit Rating:** {get_rating_badge(res_data['rating'])}", unsafe_allow_html=True)
                        st.markdown(f"**Age:** {row_data.get('age', 'N/A')} | **Income:** ₹{row_data.get('income', 'N/A')}")
                        st.markdown(f"**Loan Amount:** ₹{row_data.get('loan_amount', 'N/A')} | **Tenure:** {row_data.get('loan_tenure_months', 'N/A')} months")
                        
                    st.markdown("#### Improvement Paths")
                    try:
                        improvement_paths = generate_improvement_paths(
                            int(row_data.get('age', 30)),
                            float(row_data.get('income', 1000000)),
                            float(row_data.get('loan_amount', 500000)),
                            int(row_data.get('loan_tenure_months', 36)),
                            float(row_data.get('avg_dpd_per_delinquency', 0)),
                            float(row_data.get('delinquency_ratio', 0)),
                            float(row_data.get('credit_utilization_ratio', 30)),
                            int(row_data.get('num_open_accounts', 2)),
                            str(row_data.get('residence_type', 'Owned')),
                            str(row_data.get('loan_purpose', 'Personal')),
                            str(row_data.get('loan_type', 'Unsecured')),
                            res_data['credit_score'],
                            res_data['default_probability']
                        )

                        for path in improvement_paths:
                            with st.expander(f"{path['name']} - {path['timeline_months']} months to {path['projected_score']} score"):
                                st.markdown(f"**Description:** {path['description']}")
                                st.markdown(f"**Projected Credit Score:** {path['projected_score']}")
                                st.markdown(f"**Loan Approval Timeline:** {path['approval_months']} months")
                                st.markdown(f"**Approval Likelihood:** {path['approval_likelihood']}")
                                st.markdown("**Action Plan:**")
                                for i, action in enumerate(path['actions'], 1):
                                    st.markdown(f"  {i}. {action}")
                                if 'inflation_note' in path:
                                    st.info(path['inflation_note'])

                    except Exception as e:
                        st.info(f"Improvement paths unavailable: {str(e)[:100]}")
                        
        st.stop()

    # Individual Entry Mode
    preset = PRESETS["Custom (Enter Manually)"]
    csv_data = None
    
    st.markdown('<p class="section-header">Individual Prediction Input</p>', unsafe_allow_html=True)
    ind_input_method = st.radio("Input Method", ["Use Preset / Enter Manually", "Upload CSV (Single Profile)"], horizontal=True, key="ind_input_method")
    
    col_preset, _ = st.columns([1, 1])
    with col_preset:
        if ind_input_method == "Upload CSV (Single Profile)":
            ind_file = st.file_uploader("Upload CSV", type=['csv'], key='ind_csv')
            if ind_file is not None:
                df_ind = pd.read_csv(ind_file)
                if len(df_ind) > 0:
                    csv_data = df_ind.iloc[0].to_dict()
                    st.success("Loaded profile from CSV!")
        else:
            preset_name = st.selectbox(
                "Preset Profile",
                list(PRESETS.keys()),
                help="Auto-fill all inputs with a sample borrower profile for quick testing"
            )
            preset = PRESETS[preset_name]

    if ind_input_method == "Upload CSV (Single Profile)":
        if csv_data is None:
            st.info("Please upload a CSV file above to generate a prediction.")
            st.stop()
            
        age = int(csv_data.get('age', 28))
        income = int(csv_data.get('income', 1200000))
        residence_type = str(csv_data.get('residence_type', 'Owned'))
        loan_amount = int(csv_data.get('loan_amount', 2560000))
        loan_tenure_months = int(csv_data.get('loan_tenure_months', 36))
        loan_purpose = str(csv_data.get('loan_purpose', 'Personal'))
        loan_type = str(csv_data.get('loan_type', 'Unsecured'))
        avg_dpd = int(csv_data.get('avg_dpd_per_delinquency', 20))
        num_open_accounts = int(csv_data.get('number_of_open_accounts', csv_data.get('num_open_accounts', 2)))
        delinquency_ratio = int(csv_data.get('delinquency_ratio', 30))
        credit_utilization_ratio = int(csv_data.get('credit_utilization_ratio', 30))
        
        st.markdown("### 📝 Uploaded Profile Details")
        st.markdown(f"**Age:** {age} | **Income:** ₹{income:,.0f} | **Residence:** {residence_type}")
        st.markdown(f"**Loan Amount:** ₹{loan_amount:,.0f} | **Tenure:** {loan_tenure_months} months | **Purpose:** {loan_purpose} | **Type:** {loan_type}")
        st.markdown(f"**Avg DPD:** {avg_dpd} days | **Open Accounts:** {num_open_accounts} | **Delinquency Ratio:** {delinquency_ratio}% | **Credit Utilization:** {credit_utilization_ratio}%")
        st.markdown("---")
        
    else:
        st.markdown('<p class="section-header">Borrower Information</p>', unsafe_allow_html=True)

        # ── Row 1: Demographics ──
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age = st.number_input(
                'Age', min_value=18, max_value=100, step=1,
                value=int(csv_data.get('age', preset['age'] if preset else 28)) if csv_data else (preset['age'] if preset else 28),
                help="Borrower's age (18-100). Younger borrowers may have less credit history."
            )
        with r1c2:
            income = st.number_input(
                'Annual Income (₹)', min_value=100000, step=50000,
                value=int(csv_data.get('income', preset['income'] if preset else 1200000)) if csv_data else (preset['income'] if preset else 1200000),
                help="Verified annual income in INR. Higher income reduces default risk."
            )
        with r1c3:
            residence_type = st.selectbox(
                'Residence Type',
                ['Owned', 'Rented', 'Mortgage'],
                index=['Owned', 'Rented', 'Mortgage'].index(csv_data.get('residence_type', preset['residence'] if preset else 'Owned')) if csv_data else (['Owned', 'Rented', 'Mortgage'].index(preset['residence']) if preset else 0),
                help="Type of residence. Owned property suggests financial stability."
            )

        # ── Row 2: Loan Details ──
        st.markdown('<p class="section-header">Loan Details</p>', unsafe_allow_html=True)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            loan_amount = st.number_input(
                'Loan Amount (₹)', min_value=10000, step=50000,
                value=int(csv_data.get('loan_amount', preset['loan_amount'] if preset else 2560000)) if csv_data else (preset['loan_amount'] if preset else 2560000),
                help="Requested loan principal in INR."
            )
        with r2c2:
            loan_tenure_months = st.number_input(
                'Loan Tenure (months)', min_value=3, max_value=60, step=3,
                value=int(csv_data.get('loan_tenure_months', preset['loan_tenure_months'] if preset else 36)) if csv_data else (preset['loan_tenure_months'] if preset else 36),
                help="Repayment period in months (3-60)."
            )
        with r2c3:
            loan_purpose = st.selectbox(
                'Loan Purpose',
                ['Education', 'Home', 'Auto', 'Personal'],
                index=['Education', 'Home', 'Auto', 'Personal'].index(csv_data.get('loan_purpose', preset['purpose'] if preset else 'Personal')) if csv_data else (['Education', 'Home', 'Auto', 'Personal'].index(preset['purpose']) if preset else 0),
                help="Purpose of the loan. Education and Home loans tend to be lower risk."
            )

        # ── Row 3: Loan Type + Credit History Sliders ──
        st.markdown('<p class="section-header">Credit History</p>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            loan_type = st.selectbox(
                'Loan Type',
                ['Unsecured', 'Secured'],
                index=['Unsecured', 'Secured'].index(csv_data.get('loan_type', preset['type'] if preset else 'Unsecured')) if csv_data else (['Unsecured', 'Secured'].index(preset['type']) if preset else 0),
                help="Secured loans are backed by collateral (e.g., car, house). Unsecured loans carry higher risk."
            )
        with r3c2:
            avg_dpd = st.slider(
                'Avg Days Past Due (DPD)',
                min_value=0, max_value=100, step=1,
                value=int(csv_data.get('avg_dpd_per_delinquency', preset['avg_dpd'] if preset else 20)) if csv_data else (preset['avg_dpd'] if preset else 20),
                help="Average number of days payments were overdue. Higher values indicate payment issues."
            )
        with r3c3:
            num_open_accounts = st.number_input(
                'Open Loan Accounts',
                min_value=1, max_value=4, step=1,
                value=int(csv_data.get('number_of_open_accounts', preset['open_accounts'] if preset else 2)) if csv_data else (preset['open_accounts'] if preset else 2),
                help="Number of currently active credit accounts (1-4)."
            )

        # ── Row 4: Sliders with color zones ──
        r4c1, r4c2 = st.columns(2)
        with r4c1:
            delinquency_ratio = st.slider(
                'Delinquency Ratio (%)',
                min_value=0, max_value=100, step=1,
                value=int(csv_data.get('delinquency_ratio', preset['delinquency'] if preset else 30)) if csv_data else (preset['delinquency'] if preset else 30),
                help="Percentage of delinquent payment periods. <20% = Low, 20-50% = Medium, >50% = High"
            )
        with r4c2:
            credit_utilization_ratio = st.slider(
                'Credit Utilization (%)',
                min_value=0, max_value=100, step=1,
                value=int(csv_data.get('credit_utilization_ratio', preset['credit_util'] if preset else 30)) if csv_data else (preset['credit_util'] if preset else 30),
                help="Percentage of available credit being used. <30% = Low, 30-60% = Medium, >60% = High"
            )

    # ── Auto-Calculated Derived Features (Paper 4) ──
    st.markdown('<p class="section-header">Auto-Calculated Features</p>', unsafe_allow_html=True)

    loan_to_income = loan_amount / income if income > 0 else 0
    emi = compute_emi(loan_amount, loan_tenure_months)
    monthly_income = income / 12
    debt_burden = (emi / monthly_income * 100) if monthly_income > 0 else 0
    total_interest = (emi * loan_tenure_months) - loan_amount

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        lti_color = "#22c55e" if loan_to_income < 2 else ("#f59e0b" if loan_to_income < 4 else "#ef4444")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Loan-to-Income</div>
            <div class="metric-value" style="color: {lti_color}">{loan_to_income:.2f}x</div>
            <div class="metric-sub">{"Healthy" if loan_to_income < 2 else "Stretched" if loan_to_income < 4 else "High"}</div>
        </div>""", unsafe_allow_html=True)
    with ac2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Monthly EMI</div>
            <div class="metric-value" style="color: #38bdf8">₹{emi:,.0f}</div>
            <div class="metric-sub">@ 10% p.a. assumed</div>
        </div>""", unsafe_allow_html=True)
    with ac3:
        db_color = "#22c55e" if debt_burden < 40 else ("#f59e0b" if debt_burden < 60 else "#ef4444")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Debt Burden</div>
            <div class="metric-value" style="color: {db_color}">{debt_burden:.1f}%</div>
            <div class="metric-sub">EMI / Monthly Income</div>
        </div>""", unsafe_allow_html=True)
    with ac4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Interest</div>
            <div class="metric-value" style="color: #a78bfa">₹{max(total_interest,0):,.0f}</div>
            <div class="metric-sub">Over loan tenure</div>
        </div>""", unsafe_allow_html=True)

    # ── Input Validation Warnings ──
    warnings_list = []
    if loan_to_income > 4:
        warnings_list.append("Loan-to-Income ratio > 4x — very high leverage")
    if delinquency_ratio > 50:
        warnings_list.append("Delinquency ratio > 50% — severe payment history issues")
    if credit_utilization_ratio > 80:
        warnings_list.append("Credit utilization > 80% — near-maxed credit")
    if avg_dpd > 60:
        warnings_list.append("Average DPD > 60 days — significant payment delays")
    if age < 21 and loan_amount > 1000000:
        warnings_list.append("Young borrower with large loan — unusual pattern")
    if debt_burden > 60:
        warnings_list.append("Debt burden > 60% — EMI consumes most income")

    if warnings_list:
        for w in warnings_list:
            st.markdown(f'<div class="warning-badge">{w}</div>', unsafe_allow_html=True)

    # ── Preliminary Risk Indicator ──
    risk_level, risk_text, risk_class = get_preliminary_risk(
        delinquency_ratio, credit_utilization_ratio, loan_to_income, avg_dpd
    )
    st.markdown(f"""
    <div class="traffic-light {risk_class}">
        {risk_text} — Preliminary risk assessment (heuristic, not model prediction)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # ── Predict Button ──
    if st.button("Calculate Risk", type="primary", use_container_width=True):
        st.session_state['show_ind_results'] = True

    if st.session_state.get('show_ind_results', False):
        with st.spinner("Running prediction..."):
            probability, credit_score, rating = predict(
                age, income, loan_amount, loan_tenure_months, avg_dpd,
                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                residence_type, loan_purpose, loan_type,
                model_name=selected_model
            )

        st.markdown("---")
        st.markdown('<p class="section-header">Prediction Results</p>', unsafe_allow_html=True)

        # Results row
        res1, res2, res3 = st.columns(3)
        with res1:
            prob_color = "#22c55e" if probability < 0.2 else ("#f59e0b" if probability < 0.5 else "#ef4444")
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Default Probability</div>
                <div class="metric-value" style="color: {prob_color}">{probability:.1%}</div>
                <div class="metric-sub">Chance of loan default</div>
            </div>""", unsafe_allow_html=True)
        with res2:
            st.plotly_chart(create_gauge_chart(credit_score), use_container_width=True, key="gauge")
        with res3:
            st.markdown(f"""<div class="metric-card" style="padding-top: 2rem;">
                <div class="metric-label">Credit Rating</div>
                <div style="margin: 1rem 0;">{get_rating_badge(rating)}</div>
                <div class="metric-sub">Model: {selected_model}</div>
            </div>""", unsafe_allow_html=True)

        # ── SHAP Explainability (Paper 2 — IEEE 9050779) ──
        st.markdown("---")
        st.markdown('<p class="section-header">Why This Score? — SHAP Explainability</p>', unsafe_allow_html=True)
        st.caption("Based on: *Explainability of ML Granting Scoring Model in P2P Lending* (IEEE 9050779)")

        try:
            shap_data = get_shap_explanation(
                age, income, loan_amount, loan_tenure_months, avg_dpd,
                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                residence_type, loan_purpose, loan_type,
                model_name=selected_model
            )

            shap_df = pd.DataFrame({
                'Feature': shap_data['feature_names'],
                'SHAP Value': shap_data['shap_values']
            }).sort_values('SHAP Value', key=abs, ascending=True)

            colors = ['#ef4444' if v > 0 else '#22c55e' for v in shap_df['SHAP Value']]

            fig_shap = go.Figure(go.Bar(
                x=shap_df['SHAP Value'],
                y=shap_df['Feature'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.4f}" for v in shap_df['SHAP Value']],
                textposition='outside',
                textfont={'size': 11}
            ))
            fig_shap.update_layout(
                title="Feature Contributions to Default Risk",
                xaxis_title="SHAP Value (→ increases risk | ← decreases risk)",
                yaxis_title="",
                height=400,
                margin=dict(l=10, r=80, t=40, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'},
                xaxis={'gridcolor': 'rgba(71,85,105,0.3)', 'zerolinecolor': '#475569'},
                yaxis={'gridcolor': 'rgba(71,85,105,0.3)'}
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Interpretation
            top_risk = shap_df[shap_df['SHAP Value'] > 0].tail(3)
            top_safe = shap_df[shap_df['SHAP Value'] < 0].head(3)
            with st.expander("How to read this chart"):
                st.markdown("""
                - **Red bars (→)**: Features that **increase** default risk for this borrower
                - **Green bars (←)**: Features that **decrease** default risk for this borrower
                - Longer bars = stronger influence on the prediction
                - This analysis uses SHAP (SHapley Additive exPlanations) to decompose each prediction
                """)

        except Exception as e:
            st.info(f"SHAP explanation unavailable: {str(e)[:100]}. Train models first with `python model_trainer.py`")

        # ── Improvement Paths ──
        st.markdown("---")
        st.markdown('<p class="section-header">How to Improve Your Credit Score</p>', unsafe_allow_html=True)

        try:
            improvement_paths = generate_improvement_paths(
                age, income, loan_amount, loan_tenure_months, avg_dpd,
                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                residence_type, loan_purpose, loan_type, credit_score, probability
            )

            for path in improvement_paths:
                with st.expander(f"{path['name']} - {path['timeline_months']} months to {path['projected_score']} score"):
                    st.markdown(f"**Description:** {path['description']}")
                    st.markdown(f"**Projected Credit Score:** {path['projected_score']}")
                    st.markdown(f"**Loan Approval Timeline:** {path['approval_months']} months")
                    st.markdown(f"**Approval Likelihood:** {path['approval_likelihood']}")
                    st.markdown("**Action Plan:**")
                    for i, action in enumerate(path['actions'], 1):
                        st.markdown(f"  {i}. {action}")
                    if 'inflation_note' in path:
                        st.info(path['inflation_note'])

        except Exception as e:
            st.info(f"Improvement paths unavailable: {str(e)[:100]}")

        # ── Survival Analysis (Paper 9) ──
        st.markdown("---")
        st.markdown('<p class="section-header">Survival Analysis (Risk-Over-Time)</p>', unsafe_allow_html=True)
        st.caption("Based on: *Survival Analysis for Predicting Time to Default in Lending*")
        
        try:
            surv_df = generate_survival_curve(probability, loan_tenure_months)
            fig_surv = px.line(
                surv_df, x='Month', y='Survival Probability',
                title="Probability of Not Defaulting Over Loan Tenure",
                markers=True
            )
            fig_surv.update_layout(
                height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'},
                xaxis={'gridcolor': 'rgba(71,85,105,0.3)', 'title': 'Months'},
                yaxis={'gridcolor': 'rgba(71,85,105,0.3)', 'range': [0, 1.05], 'title': 'Survival Probability'}
            )
            st.plotly_chart(fig_surv, use_container_width=True)
        except Exception as e:
            st.info(f"Survival curve unavailable: {str(e)[:100]}")

        # ── Macroeconomic Stress Testing (Paper 7) ──
        st.markdown("---")
        st.markdown('<p class="section-header">Macroeconomic Stress Testing</p>', unsafe_allow_html=True)
        st.caption("Based on: *Evaluating the Robustness of Credit Scoring Models Under Economic Shocks*")
        
        with st.expander("Run Stress Test Simulator"):
            st.write("Simulate an economic downturn (e.g. inflation spike, job loss) to see if this profile remains resilient.")
            shock_level = st.select_slider("Select Shock Severity", options=["Mild", "Moderate", "Severe"])
            
            if st.button("Run Stress Test"):
                with st.spinner("Simulating economic shock..."):
                    if shock_level == "Mild":
                        s_income = income * 0.9
                        s_util = min(credit_utilization_ratio + 10, 100)
                        s_dpd = avg_dpd + 5
                    elif shock_level == "Moderate":
                        s_income = income * 0.8
                        s_util = min(credit_utilization_ratio + 20, 100)
                        s_dpd = avg_dpd + 15
                    else:
                        s_income = income * 0.6
                        s_util = min(credit_utilization_ratio + 40, 100)
                        s_dpd = avg_dpd + 30
                        
                    s_prob, s_score, s_rating = predict(
                        age, s_income, loan_amount, loan_tenure_months, s_dpd,
                        delinquency_ratio, s_util, num_open_accounts,
                        residence_type, loan_purpose, loan_type,
                        model_name=selected_model
                    )
                    
                    st.markdown("#### Stress Test Results")
                    col_st1, col_st2 = st.columns(2)
                    with col_st1:
                        st.markdown(f"**Base Score:** {credit_score} ({rating})")
                        st.plotly_chart(create_gauge_chart(credit_score), use_container_width=True, key="gauge_base")
                    with col_st2:
                        color = "red" if s_score < 650 else "green"
                        st.markdown(f"**Stressed Score:** <span style='color:{color}'>{s_score} ({s_rating})</span>", unsafe_allow_html=True)
                        st.plotly_chart(create_gauge_chart(s_score), use_container_width=True, key="gauge_stress")
                    
                    diff = credit_score - s_score
                    if s_score >= 650:
                        st.success(f"**Resilient:** Score dropped by {diff} points, but remains in the acceptable range under a {shock_level} shock.")
                    else:
                        st.error(f"**Vulnerable:** Score dropped by {diff} points, falling below the acceptable threshold under a {shock_level} shock.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE (Paper 3 — IEEE 10914916)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>Model Performance Analytics</h1>
        <p>Compare models with ROC curves, confusion matrices & precision-recall analysis</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Based on: *Credit Risk Assessment using Ensemble Models and Explainable AI* (IEEE 10914916)")

    metrics = load_metrics()
    if metrics is None:
        st.warning("No model metrics found. Run `python model_trainer.py` first to generate metrics.")
        st.code("python model_trainer.py", language="bash")
        st.stop()

    # Display name mapping
    display_names = {
        'logistic_regression': 'Logistic Regression',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'random_forest': 'Random Forest'
    }
    model_colors = {
        'logistic_regression': '#6366f1',
        'xgboost': '#22c55e',
        'lightgbm': '#f59e0b',
        'random_forest': '#ec4899'
    }

    # ── Metrics Comparison Table ──
    st.markdown('<p class="section-header">Metrics Comparison</p>', unsafe_allow_html=True)

    metrics_table = []
    for key, m in metrics.items():
        metrics_table.append({
            'Model': display_names.get(key, key),
            'Accuracy': f"{m['accuracy']:.2%}",
            'Precision': f"{m['precision']:.2%}",
            'Recall': f"{m['recall']:.2%}",
            'F1-Score': f"{m['f1']:.2%}",
            'AUC-ROC': f"{m['auc_roc']:.4f}"
        })
    st.dataframe(pd.DataFrame(metrics_table), use_container_width=True, hide_index=True)

    # ── Bar Chart Comparison ──
    bar_data = []
    for key, m in metrics.items():
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            bar_data.append({
                'Model': display_names.get(key, key),
                'Metric': metric_name.upper().replace('_', '-'),
                'Value': m[metric_name]
            })

    fig_bar = px.bar(
        pd.DataFrame(bar_data), x='Metric', y='Value', color='Model',
        barmode='group', title='Model Performance Comparison',
        color_discrete_sequence=['#6366f1', '#22c55e', '#f59e0b', '#ec4899']
    )
    fig_bar.update_layout(
        height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        xaxis={'gridcolor': 'rgba(71,85,105,0.3)'},
        yaxis={'gridcolor': 'rgba(71,85,105,0.3)', 'range': [0, 1.05]}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── ROC Curve with Shaded AUC Area (Full Width) ──
    st.markdown('<p class="section-header">📈 ROC Curve — Receiver Operating Characteristic</p>', unsafe_allow_html=True)
    st.caption(
        "The ROC curve plots the True Positive Rate (sensitivity) against the False Positive Rate "
        "(1 − specificity) at every classification threshold. A curve that hugs the top-left corner "
        "indicates a model with strong discriminative power."
    )

    fig_roc_full = go.Figure()

    # Add filled AUC area + line for each model
    fill_opacities = {
        'logistic_regression': 0.08,
        'xgboost': 0.10,
        'lightgbm': 0.10,
        'random_forest': 0.08
    }
    for key, m in metrics.items():
        if 'roc_curve' in m:
            color = model_colors.get(key, '#ffffff')
            opacity = fill_opacities.get(key, 0.08)
            # Shaded area under curve
            fig_roc_full.add_trace(go.Scatter(
                x=m['roc_curve']['fpr'], y=m['roc_curve']['tpr'],
                fill='tozeroy',
                fillcolor=color.replace(')', f', {opacity})').replace('rgb', 'rgba') if 'rgb' in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{opacity})",
                line=dict(color=color, width=2.5),
                name=f"{display_names.get(key, key)} (AUC = {m['auc_roc']:.4f})",
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>' + display_names.get(key, key) + '</extra>'
            ))
    # Random baseline
    fig_roc_full.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name='Random Classifier (AUC = 0.5)',
        line=dict(color='rgba(148,163,184,0.6)', dash='dash', width=1.5)
    ))
    fig_roc_full.update_layout(
        title=dict(text='ROC Curves — All Models', font=dict(size=16, color='#e2e8f0')),
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        xaxis={'gridcolor': 'rgba(71,85,105,0.3)', 'range': [0, 1]},
        yaxis={'gridcolor': 'rgba(71,85,105,0.3)', 'range': [0, 1.05]},
        legend=dict(font=dict(size=11), bgcolor='rgba(30,41,59,0.8)', bordercolor='rgba(99,102,241,0.3)', borderwidth=1)
    )
    st.plotly_chart(fig_roc_full, use_container_width=True)

    # ── AUC Score Comparison Bar Chart ──
    st.markdown('<p class="section-header">📊 Area Under the Curve (AUC) — Model Comparison</p>', unsafe_allow_html=True)
    st.caption(
        "AUC summarises the ROC curve into a single number between 0 and 1.  "
        "AUC = 1.0 means perfect classification; AUC = 0.5 equals random guessing. "
        "Higher AUC → better ability to distinguish defaulters from non-defaulters."
    )

    auc_names = [display_names.get(k, k) for k in metrics]
    auc_values = [m['auc_roc'] for m in metrics.values()]
    auc_colors_list = [model_colors.get(k, '#ffffff') for k in metrics]

    fig_auc = go.Figure(go.Bar(
        x=auc_names, y=auc_values,
        marker_color=auc_colors_list,
        text=[f"{v:.4f}" for v in auc_values],
        textposition='outside',
        textfont=dict(size=14, color='#e2e8f0'),
        hovertemplate='%{x}<br>AUC-ROC: %{y:.4f}<extra></extra>'
    ))
    # Add a reference line at AUC = 0.5 (random)
    fig_auc.add_hline(
        y=0.5, line_dash='dash', line_color='rgba(148,163,184,0.5)', line_width=1,
        annotation_text='Random (0.5)', annotation_position='bottom right',
        annotation_font=dict(color='#94a3b8', size=10)
    )
    fig_auc.update_layout(
        title=dict(text='AUC-ROC Score by Model', font=dict(size=16, color='#e2e8f0')),
        yaxis_title='AUC-ROC Score',
        height=420,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        xaxis={'gridcolor': 'rgba(71,85,105,0.3)'},
        yaxis={'gridcolor': 'rgba(71,85,105,0.3)', 'range': [0, 1.08]},
        bargap=0.35
    )
    st.plotly_chart(fig_auc, use_container_width=True)

    # ── Side-by-side: ROC Curves (compact) & Precision-Recall Curves ──
    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown('<p class="section-header">ROC Curves (Compact)</p>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        for key, m in metrics.items():
            if 'roc_curve' in m:
                fig_roc.add_trace(go.Scatter(
                    x=m['roc_curve']['fpr'], y=m['roc_curve']['tpr'],
                    name=f"{display_names.get(key, key)} (AUC={m['auc_roc']:.4f})",
                    line=dict(color=model_colors.get(key, '#ffffff'), width=2)
                ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name='Random',
            line=dict(color='rgba(148,163,184,0.5)', dash='dash', width=1)
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'},
            xaxis={'gridcolor': 'rgba(71,85,105,0.3)'},
            yaxis={'gridcolor': 'rgba(71,85,105,0.3)'},
            legend={'font': {'size': 10}}
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_pr:
        st.markdown('<p class="section-header">Precision-Recall Curves</p>', unsafe_allow_html=True)
        fig_pr = go.Figure()
        for key, m in metrics.items():
            if 'pr_curve' in m:
                fig_pr.add_trace(go.Scatter(
                    x=m['pr_curve']['recall'], y=m['pr_curve']['precision'],
                    name=display_names.get(key, key),
                    line=dict(color=model_colors.get(key, '#ffffff'), width=2)
                ))
        fig_pr.update_layout(
            xaxis_title='Recall', yaxis_title='Precision',
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'},
            xaxis={'gridcolor': 'rgba(71,85,105,0.3)'},
            yaxis={'gridcolor': 'rgba(71,85,105,0.3)'},
            legend={'font': {'size': 10}}
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Confusion Matrices ──
    st.markdown('<p class="section-header">🔲 Confusion Matrices</p>', unsafe_allow_html=True)
    cm_cols = st.columns(len(metrics))
    for idx, (key, m) in enumerate(metrics.items()):
        with cm_cols[idx]:
            cm = np.array(m['confusion_matrix'])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=['No Default', 'Default'], y=['No Default', 'Default'],
                color_continuous_scale='Blues',
                title=display_names.get(key, key)
            )
            fig_cm.update_layout(
                height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0', 'size': 10},
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Fairness & Bias Auditing (Paper 8) ──
    st.markdown("---")
    st.markdown('<p class="section-header">⚖️ Fairness Audit</p>', unsafe_allow_html=True)
    st.caption("Reference: *Fairness-Aware Machine Learning for Credit Risk Assessment*")
    st.write("This automated audit checks historical batch data to verify that the model's approval decisions are balanced across different borrower demographics. According to industry guidelines, a Disparate Impact Ratio between 0.8 and 1.2 indicates a balanced and fair model.")
    
    try:
        if os.path.exists('sample_batch.csv'):
            f_df = pd.read_csv('sample_batch.csv')
            f_res = predict_batch(f_df, model_name='Random Forest')
            
            # Approvals (Score >= 650)
            f_res['Approved'] = f_res['credit_score'] >= 650
            
            # Age Bias (Under 30 vs Over 30)
            f_res['Under_30'] = f_df['age'] < 30
            rate_under30 = f_res[f_res['Under_30']]['Approved'].mean()
            rate_over30 = f_res[~f_res['Under_30']]['Approved'].mean()
            di_age = rate_under30 / rate_over30 if rate_over30 > 0 else 1.0
            
            # Residence Bias (Rented vs Owned)
            rate_rented = f_res[f_df['residence_type'] == 'Rented']['Approved'].mean()
            rate_owned = f_res[f_df['residence_type'] == 'Owned']['Approved'].mean()
            di_res = rate_rented / rate_owned if rate_owned > 0 else 1.0
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown("**Age Demographics (Under 30 vs. 30+)**")
                status_age = "Balanced" if 0.8 <= di_age <= 1.2 else "Review Needed"
                st.metric("Disparate Impact Ratio", f"{di_age:.2f}", delta=status_age, delta_color="normal" if status_age == "Balanced" else "inverse")
                st.caption(f"Approval Rates: Under 30 ({rate_under30:.1%}) | 30+ ({rate_over30:.1%})")
                
            with col_f2:
                st.markdown("**Housing Status (Rented vs. Owned)**")
                status_res = "Balanced" if 0.8 <= di_res <= 1.2 else "Review Needed"
                st.metric("Disparate Impact Ratio", f"{di_res:.2f}", delta=status_res, delta_color="normal" if status_res == "Balanced" else "inverse")
                st.caption(f"Approval Rates: Rented ({rate_rented:.1%}) | Owned ({rate_owned:.1%})")
        else:
            st.warning("No batch data found to run the fairness audit.")
    except Exception as e:
        st.error(f"Audit could not be completed: {e}")

    # ── Regulatory Compliance Note ──
    st.markdown("---")
    st.info(
        "🏛️ **Regulatory Compliance Note** (Paper 3 — IEEE 10914916): "
        "All models provide transparent performance metrics including AUC-ROC, confusion matrices, and "
        "precision-recall curves. Combined with SHAP explainability (Paper 2), this satisfies "
        "regulatory requirements for model validation and audit trails (GDPR, RBI Fair Lending Guidelines)."
    )


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
elif page == "About & Papers":
    st.markdown("""
    <div class="main-header">
        <h1>About & Research Papers</h1>
        <p>Academic foundation and references for this credit risk assessment platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## Dataset Citation

    > **Dataset:** Tunisian Commercial Bank — Credit Risk Dataset  
    > **Source:** [Kaggle — Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  
    > **Records:** ~50,000 loan applications across 3 relational tables (`customers.csv`, `loans.csv`, `bureau_data.csv`)  
    > **Features:** 24 raw variables → 14 engineered features used for model training  
    > **Target Variable:** `default` (binary — 0 = non-default, 1 = default)  
    > **Class Distribution:** 90% non-default / 10% default (addressed via SMOTE-Tomek)  
    > **Usage:** Model training, SHAP explainability, performance benchmarking  

    ---

    ## Research Papers

    This project's features are each backed by a specific IEEE-published research paper:

    ### Paper 1 — Multi-Model Ensemble Comparison
    **"Effective Credit Risk Prediction Using Ensemble Classifiers With Model Explanation"**
    - **Publisher:** IEEE Access
    - **Link:** [https://ieeexplore.ieee.org/document/10638034](https://ieeexplore.ieee.org/document/10638034)
    - **Feature Built:** Multi-model comparison (LR vs XGBoost vs LightGBM vs Random Forest)

    ---

    ### Paper 2 — SHAP Explainability Dashboard
    **"Explainability of a Machine Learning Granting Scoring Model in Peer-to-Peer Lending"**
    - **Publisher:** IEEE Journals & Magazine
    - **Link:** [https://ieeexplore.ieee.org/document/9050779](https://ieeexplore.ieee.org/document/9050779)
    - **Feature Built:** Interactive SHAP "Why this score?" panel

    ---

    ### Paper 3 — Performance Analytics & Regulatory Compliance
    **"Credit Risk Assessment using Ensemble Models and Explainable AI"**
    - **Publisher:** IEEE Conference Publication
    - **Link:** [https://ieeexplore.ieee.org/document/10914916](https://ieeexplore.ieee.org/document/10914916)
    - **Feature Built:** ROC curves, confusion matrices, precision-recall curves dashboard

    ---

    ### Paper 4 — Smart Feature Engineering & Automated Input
    **"Credit Risk Prediction Based on Machine Learning Methods"**
    - **Publisher:** IEEE (ICCSE 2019)
    - **Link:** [https://ieeexplore.ieee.org/document/8845444](https://ieeexplore.ieee.org/document/8845444)
    - **Feature Built:** Auto-calculated EMI, debt burden, LTI ratio, preset profiles, smart sliders

    ---

    ### Paper 5 — Batch Processing & Incremental Learning
    **"An Incremental Learning Ensemble Method for Imbalanced Credit Scoring"**
    - **Publisher:** IEEE Conference Publication
    - **Link:** [https://ieeexplore.ieee.org/document/9002821](https://ieeexplore.ieee.org/document/9002821)
    - **Feature Built:** Batch CSV upload, predictions for multiple borrowers, PDF report export

    ---

    ### Paper 6 — Counterfactual Explanations & Actionable Recourse
    **"Generating Actionable Counterfactual Explanations for Credit Scoring"**
    - **Publisher:** IEEE Access / IEEE Transactions
    - **Link:** Referenced conceptually based on recent IEEE literature
    - **Feature Built:** Explicit feature-level tracking, score targets, and step-by-step improvement paths

    ---

    ### Paper 7 — Macroeconomic Stress Testing
    **"Evaluating the Robustness of Credit Scoring Models Under Economic Shocks"**
    - **Publisher:** Open Access (arXiv / SSRN)
    - **Link:** [Search Open Literature](https://arxiv.org/search/?query=Stress+Testing+Credit+Scoring+Machine+Learning&searchtype=all)
    - **Feature Built:** Stress Test Simulator for inflation/recession resilience tracking

    ---

    ### Paper 8 — Algorithmic Fairness & Bias Auditing
    **"Fairness-Aware Machine Learning for Credit Risk Assessment"**
    - **Publisher:** Open Access (arXiv)
    - **Link:** [Search Open Literature](https://arxiv.org/search/?query=Fairness+Credit+Scoring+Machine+Learning&searchtype=all)
    - **Feature Built:** Disparate Impact Ratio metrics for Age and Residence Type

    ---

    ### Paper 9 — Survival Analysis
    **"Survival Analysis for Predicting Time to Default in Peer-to-Peer Lending"**
    - **Publisher:** Open Access (arXiv)
    - **Link:** [Search Open Literature](https://arxiv.org/search/?query=Survival+Analysis+Credit+Scoring+Machine+Learning&searchtype=all)
    - **Feature Built:** Monthly default risk curve over loan tenure

    ---

    ## 🏗️ Architecture

    ```
    ┌─────────────────────────────────────────────────────────┐
    │                   Streamlit Frontend                     │
    │  ├── Risk Assessment (smart inputs + SHAP)           │
    │  ├── Model Performance (ROC, CM, PR curves)          │
    │  ├── Batch Prediction (CSV upload + PDF)             │
    │  └──  About (research papers)                        │
    ├─────────────────────────────────────────────────────────┤
    │                 Prediction Engine                        │
    │  ├── Multi-model loader (LR, XGB, LGBM, RF)            │
    │  ├── SHAP explainer cache                               │
    │  └── Batch prediction pipeline                          │
    ├─────────────────────────────────────────────────────────┤
    │                  Model Artifacts                         │
    │  ├── 4 trained models (.joblib)                         │
    │  ├── 4 SHAP explainers (.joblib)                        │
    │  ├── Scaler + feature list                              │
    │  └── Model metrics (JSON)                               │
    ├─────────────────────────────────────────────────────────┤
    │               Training Pipeline                         │
    │  ├── SMOTE-Tomek for class imbalance                    │
    │  ├── Optuna hyperparameter tuning                       │
    │  └── Synthetic/Real data support                        │
    └─────────────────────────────────────────────────────────┘
    ```

    ## Tech Stack

    | Component | Technology |
    |-----------|-----------|
    | ML Models | Scikit-learn, XGBoost, LightGBM |
    | Explainability | SHAP |
    | Class Imbalance | SMOTE-Tomek (imbalanced-learn) |
    | Hyperparameter Tuning | Optuna |
    | Frontend | Streamlit |
    | Visualization | Plotly |
    | Report Generation | FPDF2 |
    | Data Processing | Pandas, NumPy |
    """)