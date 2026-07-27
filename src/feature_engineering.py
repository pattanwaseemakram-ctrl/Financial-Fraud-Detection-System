import pandas as pd
import numpy as np

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Feature Engineering from Transaction Patterns
    Extracts time-based, behavioral, geographic, and risk category features.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['time_of_day_risk'] = df['hour'].apply(lambda h: 1 if h in [0, 1, 2, 3, 4, 5, 22, 23] else 0)
    
    # Merchant category risk flags
    high_risk_merchants = ['Crypto', 'Wire_Transfer', 'Gaming', 'Electronics']
    df['high_risk_merchant'] = df['merchant_category'].apply(lambda m: 1 if m in high_risk_merchants else 0)
    
    # Geographic anomaly indicator
    suspicious_locations = ['Lagos, NG', 'Moscow, RU', 'Sydney, AU']
    df['unusual_location'] = df['location'].apply(lambda loc: 1 if loc in suspicious_locations else 0)
    
    # Behavioral features: User spending deviation from mean
    user_avg_amount = df.groupby('user_id')['amount'].transform('mean')
    df['spending_deviation'] = (df['amount'] - user_avg_amount).clip(lower=0)
    
    # 24h frequency estimate
    user_counts = df.groupby('user_id')['transaction_id'].transform('count')
    df['transaction_frequency_24h'] = user_counts.clip(lower=1, upper=15)
    
    # One-Hot Encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=['device_type', 'merchant_category'], drop_first=False)
    return df_encoded
