import pandas as pd
import numpy as np

def generate_synthetic_data(num_rows=2000):
    """
    Generates a synthetic dataset for MSME underwriting.
    """
    np.random.seed(42)
    data = {
        'business_vintage_months': np.random.randint(6, 120, num_rows),
        'industry_type': np.random.choice(
            ['Retail', 'Service', 'Manufacturing'], 
            num_rows, 
            p=[0.4, 0.4, 0.2]
        ),
        'avg_monthly_gst_turnover': np.random.normal(300000, 100000, num_rows),
        'avg_monthly_bank_balance': np.random.normal(150000, 50000, num_rows),
        'num_cheque_bounces_last_6mo': np.random.randint(0, 5, num_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Create the target variable 'loan_defaulted'
    # More bounces, lower balance, and lower turnover = higher default risk
    score = (
        - (df['avg_monthly_gst_turnover'] / 100000)
        - (df['avg_monthly_bank_balance'] / 50000)
        + (df['num_cheque_bounces_last_6mo'] * 2)
        - (df['business_vintage_months'] / 12)
    )
    
    prob = 1 / (1 + np.exp(-score / 5))  # Sigmoid function
    df['loan_defaulted'] = (prob > np.percentile(prob, 85)).astype(int) # ~15% default rate
    
    # Clean up negative values
    df['avg_monthly_gst_turnover'] = df['avg_monthly_gst_turnover'].clip(lower=0)
    df['avg_monthly_bank_balance'] = df['avg_monthly_bank_balance'].clip(lower=0)
    
    print(f"Generated {num_rows} samples.")
    print(f"Default rate: {df['loan_defaulted'].mean():.2%}")
    
    return df

if __name__ == "__main__":
    # This lets you run 'python src/credi_synth_ai/data.py' to test
    data = generate_synthetic_data()
    print(data.head())
    data.to_csv("sample_msme_data.csv", index=False)