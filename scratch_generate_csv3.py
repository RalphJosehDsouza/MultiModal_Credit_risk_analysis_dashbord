import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from prediction_helper import predict

np.random.seed(42)

buckets = {
    (300, 400): [],
    (400, 500): [],
    (500, 600): [],
    (600, 700): [],
    (700, 800): [],
    (800, 901): []
}

def get_bucket(score):
    for b_range in buckets.keys():
        if b_range[0] <= score < b_range[1]:
            return b_range
    return None

def all_buckets_full():
    return all(len(v) >= 10 for v in buckets.values())

def generate_biased_profile():
    # To get a wide range, randomize aggressively with different tiers
    ptype = np.random.choice(['excellent', 'good', 'average', 'poor', 'very_poor', 'random'])
    if ptype == 'excellent':
        return {
            'age': np.random.randint(35, 65),
            'income': np.random.randint(2000000, 5000000),
            'loan_amount': np.random.randint(100000, 1000000),
            'loan_tenure_months': np.random.choice([12, 24]),
            'avg_dpd_per_delinquency': 0,
            'delinquency_ratio': 0,
            'credit_utilization_ratio': np.random.randint(0, 10),
            'num_open_accounts': np.random.randint(1, 3),
            'residence_type': np.random.choice(['Owned']),
            'loan_purpose': np.random.choice(['Education', 'Home']),
            'loan_type': np.random.choice(['Secured'])
        }
    elif ptype == 'good':
        return {
            'age': np.random.randint(28, 55),
            'income': np.random.randint(1000000, 3000000),
            'loan_amount': np.random.randint(500000, 2000000),
            'loan_tenure_months': np.random.choice([24, 36]),
            'avg_dpd_per_delinquency': np.random.randint(0, 10),
            'delinquency_ratio': np.random.randint(0, 15),
            'credit_utilization_ratio': np.random.randint(10, 30),
            'num_open_accounts': np.random.randint(2, 4),
            'residence_type': np.random.choice(['Owned', 'Mortgage']),
            'loan_purpose': np.random.choice(['Home', 'Auto']),
            'loan_type': np.random.choice(['Secured', 'Unsecured'])
        }
    elif ptype == 'average':
        return {
            'age': np.random.randint(22, 45),
            'income': np.random.randint(400000, 1000000),
            'loan_amount': np.random.randint(500000, 2500000),
            'loan_tenure_months': np.random.choice([36, 48, 60]),
            'avg_dpd_per_delinquency': np.random.randint(15, 35),
            'delinquency_ratio': np.random.randint(25, 45),
            'credit_utilization_ratio': np.random.randint(30, 60),
            'num_open_accounts': np.random.randint(2, 5),
            'residence_type': np.random.choice(['Rented']),
            'loan_purpose': np.random.choice(['Auto', 'Personal']),
            'loan_type': np.random.choice(['Unsecured'])
        }
    elif ptype == 'poor':
        return {
            'age': np.random.randint(18, 40),
            'income': np.random.randint(200000, 600000),
            'loan_amount': np.random.randint(1500000, 3000000),
            'loan_tenure_months': np.random.choice([48, 60]),
            'avg_dpd_per_delinquency': np.random.randint(40, 70),
            'delinquency_ratio': np.random.randint(50, 80),
            'credit_utilization_ratio': np.random.randint(60, 85),
            'num_open_accounts': np.random.randint(3, 6),
            'residence_type': np.random.choice(['Rented']),
            'loan_purpose': np.random.choice(['Personal']),
            'loan_type': np.random.choice(['Unsecured'])
        }
    elif ptype == 'very_poor':
        return {
            'age': np.random.randint(18, 25),
            'income': np.random.randint(100000, 300000),
            'loan_amount': np.random.randint(3000000, 5000000),
            'loan_tenure_months': np.random.choice([60]),
            'avg_dpd_per_delinquency': np.random.randint(70, 100),
            'delinquency_ratio': np.random.randint(80, 100),
            'credit_utilization_ratio': np.random.randint(85, 100),
            'num_open_accounts': np.random.randint(4, 6),
            'residence_type': np.random.choice(['Rented']),
            'loan_purpose': np.random.choice(['Personal']),
            'loan_type': np.random.choice(['Unsecured'])
        }
    else:
        return {
            'age': np.random.randint(18, 70),
            'income': np.random.randint(200000, 5000000),
            'loan_amount': np.random.randint(100000, 5000000),
            'loan_tenure_months': np.random.choice([12, 24, 36, 48, 60]),
            'avg_dpd_per_delinquency': np.random.randint(0, 100),
            'delinquency_ratio': np.random.randint(0, 100),
            'credit_utilization_ratio': np.random.randint(0, 100),
            'num_open_accounts': np.random.randint(1, 6),
            'residence_type': np.random.choice(['Owned', 'Rented', 'Mortgage']),
            'loan_purpose': np.random.choice(['Education', 'Home', 'Auto', 'Personal']),
            'loan_type': np.random.choice(['Unsecured', 'Secured'])
        }

attempts = 0
while not all_buckets_full() and attempts < 10000:
    profile = generate_biased_profile()
    try:
        prob, score, rating = predict(
            age=profile['age'],
            income=profile['income'],
            loan_amount=profile['loan_amount'],
            loan_tenure_months=profile['loan_tenure_months'],
            avg_dpd_per_delinquency=profile['avg_dpd_per_delinquency'],
            delinquency_ratio=profile['delinquency_ratio'],
            credit_utilization_ratio=profile['credit_utilization_ratio'],
            num_open_accounts=profile['num_open_accounts'],
            residence_type=profile['residence_type'],
            loan_purpose=profile['loan_purpose'],
            loan_type=profile['loan_type'],
            model_name='Logistic Regression'
        )
        
        bucket = get_bucket(score)
        if bucket is not None and len(buckets[bucket]) < 10:
            buckets[bucket].append(profile)
    except Exception as e:
        pass
        
    attempts += 1

all_profiles = []
for b_range, profiles in buckets.items():
    if len(profiles) < 10:
        print(f"Warning: Bucket {b_range} only has {len(profiles)} samples.")
    all_profiles.extend(profiles[:10])

df = pd.DataFrame(all_profiles)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('batch_data_50.csv', index=False)

print(f"CSV generated with stratified scores. Total rows: {len(df)}")
for b, profiles in buckets.items():
    print(f"Bucket {b}: {len(profiles)}")
