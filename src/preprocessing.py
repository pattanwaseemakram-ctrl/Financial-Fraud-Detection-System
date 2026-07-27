import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.feature_engineering import extract_features

def load_and_preprocess_data(data_path: str, test_size: float = 0.25, random_state: int = 42):
    """
    Step 2: Data Cleaning and Class Imbalance Handling using SMOTE.
    Supports both synthetic logs and large PaySim datasets.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    print(f"[INFO] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Sample down large datasets (e.g. 1M+ rows) for high-performance training
    if len(df) > 50000:
        print(f"[INFO] Dataset has {len(df):,} rows. Sampling 50,000 balanced records for fast training...")
        target_col = 'isFraud' if 'isFraud' in df.columns else 'is_fraud'
        df_fraud = df[df[target_col] == 1]
        df_legit = df[df[target_col] == 0].sample(n=min(50000 - len(df_fraud), len(df[df[target_col] == 0])), random_state=random_state)
        df = pd.concat([df_fraud, df_legit]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
    df_feat = extract_features(df)
    
    drop_cols = ['transaction_id', 'user_id', 'timestamp', 'location', 'is_fraud', 'isFraud', 'isFlaggedFraud', 'step', 'nameOrig', 'nameDest', 'type']
    feature_cols = [c for c in df_feat.columns if c not in drop_cols and np.issubdtype(df_feat[c].dtype, np.number)]
    
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['is_fraud']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE Oversampling for Imbalance Handling
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    return {
        "X_train_resampled": X_train_resampled,
        "y_train_resampled": y_train_resampled,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols
    }
