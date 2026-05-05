import pandas as pd
import numpy as np

np.random.seed(42)
n_per_group = 13  # 13 * 4 = 52 entries

# Profile 1: Excellent Credit (Low Risk)
excellent = {
    'age': np.random.randint(35, 65, n_per_group),
    'income': np.random.randint(1500000, 5000000, n_per_group),
    'loan_amount': np.random.randint(500000, 2000000, n_per_group),
    'loan_tenure_months': np.random.choice([12, 24, 36], n_per_group),
    'avg_dpd_per_delinquency': np.random.randint(0, 5, n_per_group),
    'delinquency_ratio': np.random.randint(0, 10, n_per_group),
    'credit_utilization_ratio': np.random.randint(5, 25, n_per_group),
    'num_open_accounts': np.random.randint(1, 3, n_per_group),
    'residence_type': np.random.choice(['Owned', 'Mortgage'], n_per_group),
    'loan_purpose': np.random.choice(['Education', 'Home'], n_per_group),
    'loan_type': np.random.choice(['Secured'], n_per_group)
}

# Profile 2: Good Credit (Medium-Low Risk)
good = {
    'age': np.random.randint(28, 55, n_per_group),
    'income': np.random.randint(800000, 2000000, n_per_group),
    'loan_amount': np.random.randint(1000000, 3000000, n_per_group),
    'loan_tenure_months': np.random.choice([24, 36, 48], n_per_group),
    'avg_dpd_per_delinquency': np.random.randint(5, 15, n_per_group),
    'delinquency_ratio': np.random.randint(10, 25, n_per_group),
    'credit_utilization_ratio': np.random.randint(25, 45, n_per_group),
    'num_open_accounts': np.random.randint(2, 4, n_per_group),
    'residence_type': np.random.choice(['Owned', 'Rented'], n_per_group),
    'loan_purpose': np.random.choice(['Home', 'Auto', 'Personal'], n_per_group),
    'loan_type': np.random.choice(['Secured', 'Unsecured'], n_per_group)
}

# Profile 3: Average Credit (Medium-High Risk)
average = {
    'age': np.random.randint(22, 45, n_per_group),
    'income': np.random.randint(400000, 1000000, n_per_group),
    'loan_amount': np.random.randint(500000, 2500000, n_per_group),
    'loan_tenure_months': np.random.choice([36, 48, 60], n_per_group),
    'avg_dpd_per_delinquency': np.random.randint(15, 35, n_per_group),
    'delinquency_ratio': np.random.randint(25, 45, n_per_group),
    'credit_utilization_ratio': np.random.randint(45, 75, n_per_group),
    'num_open_accounts': np.random.randint(2, 5, n_per_group),
    'residence_type': np.random.choice(['Rented'], n_per_group),
    'loan_purpose': np.random.choice(['Auto', 'Personal'], n_per_group),
    'loan_type': np.random.choice(['Unsecured'], n_per_group)
}

# Profile 4: Poor Credit (High Risk)
poor = {
    'age': np.random.randint(18, 40, n_per_group),
    'income': np.random.randint(200000, 600000, n_per_group),
    'loan_amount': np.random.randint(1500000, 4000000, n_per_group),
    'loan_tenure_months': np.random.choice([48, 60], n_per_group),
    'avg_dpd_per_delinquency': np.random.randint(40, 85, n_per_group),
    'delinquency_ratio': np.random.randint(50, 95, n_per_group),
    'credit_utilization_ratio': np.random.randint(75, 95, n_per_group),
    'num_open_accounts': np.random.randint(3, 6, n_per_group),
    'residence_type': np.random.choice(['Rented'], n_per_group),
    'loan_purpose': np.random.choice(['Personal'], n_per_group),
    'loan_type': np.random.choice(['Unsecured'], n_per_group)
}

df_exc = pd.DataFrame(excellent)
df_good = pd.DataFrame(good)
df_avg = pd.DataFrame(average)
df_poor = pd.DataFrame(poor)

df = pd.concat([df_exc, df_good, df_avg, df_poor], ignore_index=True)
# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('batch_data_50.csv', index=False)
print("CSV generated with a variety of profiles successfully.")
