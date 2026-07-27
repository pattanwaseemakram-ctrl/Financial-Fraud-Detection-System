import pandas as pd
import numpy as np

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Feature Engineering supporting both custom synthetic logs and PaySim datasets.
    """
    df = df.copy()
    
    # Column mapping for PaySim dataset (e.g. dataset fraud.csv)
    if 'isFraud' in df.columns:
        df['is_fraud'] = df['isFraud']
    if 'type' in df.columns and 'merchant_category' not in df.columns:
        df['merchant_category'] = df['type']
    if 'nameOrig' in df.columns and 'user_id' not in df.columns:
        df['user_id'] = df['nameOrig']
    if 'nameDest' in df.columns and 'location' not in df.columns:
        df['location'] = df['nameDest'].apply(lambda x: 'Online' if str(x).startswith('M') else 'Transfer_Acc')
    if 'device_type' not in df.columns:
        df['device_type'] = 'Mobile'
        
    if 'step' in df.columns and 'timestamp' not in df.columns:
        df['hour'] = df['step'] % 24
        df['day_of_week'] = (df['step'] // 24) % 7
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    else:
        df['hour'] = 12
        df['day_of_week'] = 1
        
    df['time_of_day_risk'] = df['hour'].apply(lambda h: 1 if h in [0, 1, 2, 3, 4, 5, 22, 23] else 0)
    
    # High-risk categories (includes TRANSFER & CASH_OUT from PaySim)
    high_risk_merchants = ['TRANSFER', 'CASH_OUT', 'Crypto', 'Wire_Transfer', 'Gaming', 'Electronics']
    df['high_risk_merchant'] = df['merchant_category'].apply(lambda m: 1 if m in high_risk_merchants else 0)
    
    # Geographic anomaly indicator
    suspicious_locations = ['Lagos, NG', 'Moscow, RU', 'Sydney, AU', 'Transfer_Acc']
    df['unusual_location'] = df['location'].apply(lambda loc: 1 if loc in suspicious_locations else 0)
    
    # Behavioral features
    user_avg_amount = df.groupby('user_id')['amount'].transform('mean')
    df['spending_deviation'] = (df['amount'] - user_avg_amount).clip(lower=0)
    
    user_counts = df.groupby('user_id')['amount'].transform('count')
    df['transaction_frequency_24h'] = user_counts.clip(lower=1, upper=15)
    
    # Balance discrepancy features (if present)
    if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    
    # One-Hot Encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=['device_type', 'merchant_category'], drop_first=False)
    return df_encoded
