import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

class FraudScoringEngine:
    """
    Step 6 & 8: Fraud Risk Scoring Engine & Alert Mechanism
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.load_artifacts()
        
    def load_artifacts(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, 'r') as f:
                    self.metrics = json.load(f)
            print("FraudScoringEngine: Model artifacts loaded successfully.")
        else:
            print("FraudScoringEngine WARNING: Artifacts not found. Run src/train_model.py first.")

    def analyze_transaction(self, tx_dict: dict) -> dict:
        amount = float(tx_dict.get("amount", 0.0))
        merchant = tx_dict.get("merchant_category", "Retail")
        location = tx_dict.get("location", "Online")
        device = tx_dict.get("device_type", "Mobile")
        tx_time_raw = tx_dict.get("timestamp")
        if not tx_time_raw:
            tx_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            tx_time_str = str(tx_time_raw)
            
        try:
            dt = datetime.strptime(tx_time_str, "%Y-%m-%d %H:%M:%S")
            hour = dt.hour
        except Exception:
            dt = datetime.now()
            hour = dt.hour
            
        time_of_day_risk = 1 if hour in [0, 1, 2, 3, 4, 5, 22, 23] else 0
        high_risk_merchants = ['Crypto', 'Wire_Transfer', 'Gaming', 'Electronics']
        high_risk_merchant = 1 if merchant in high_risk_merchants else 0
        suspicious_locations = ['Lagos, NG', 'Moscow, RU', 'Sydney, AU']
        unusual_location = 1 if location in suspicious_locations else 0
        
        user_avg = float(tx_dict.get("user_avg_amount", 150.0))
        spending_deviation = max(0.0, amount - user_avg)
        tx_freq = int(tx_dict.get("transaction_frequency_24h", 1))
        
        # Build audit flag list
        audit_flags = []
        if amount > 2000:
            audit_flags.append(f"Unusually High Transaction Amount (${amount:,.2f})")
        if time_of_day_risk == 1:
            audit_flags.append(f"Off-Hours Activity ({hour:02d}:00 HRS)")
        if high_risk_merchant == 1:
            audit_flags.append(f"High-Risk Category ({merchant})")
        if unusual_location == 1:
            audit_flags.append(f"Suspicious Geographic Location ({location})")
        if spending_deviation > 500:
            audit_flags.append(f"Significant Spending Spike (+${spending_deviation:,.2f} over user avg)")
        if tx_freq > 5:
            audit_flags.append(f"High 24h Frequency Velocity ({tx_freq} txns)")

        # Compute probability using trained ML model
        if self.model and self.scaler:
            feature_cols = self.metrics.get("feature_columns", [])
            row_dict = {
                "amount": amount,
                "hour": hour,
                "day_of_week": dt.weekday(),
                "time_of_day_risk": time_of_day_risk,
                "high_risk_merchant": high_risk_merchant,
                "unusual_location": unusual_location,
                "spending_deviation": spending_deviation,
                "transaction_frequency_24h": tx_freq
            }
            
            for d in ["ATM", "Desktop", "Mobile", "POS"]:
                row_dict[f"device_type_{d}"] = 1 if device == d else 0
                
            for m in ["Crypto", "Electronics", "Gaming", "Grocery", "Retail", "Travel", "Wire_Transfer"]:
                row_dict[f"merchant_category_{m}"] = 1 if merchant == m else 0
                
            df_row = pd.DataFrame([row_dict])
            for col in feature_cols:
                if col not in df_row.columns:
                    df_row[col] = 0
            df_row = df_row[feature_cols]
            
            scaled_row = self.scaler.transform(df_row)
            prob_fraud = float(self.model.predict_proba(scaled_row)[0][1])
            threshold = self.metrics.get("optimal_threshold", 0.35)
        else:
            prob_fraud = 0.50
            threshold = 0.35
            
        risk_score_pct = round(prob_fraud * 100, 1)
        
        # Step 8: Action Routing & Risk Level Determination
        if risk_score_pct >= 70.0 or (prob_fraud >= threshold and len(audit_flags) >= 2):
            risk_level = "HIGH"
            action_taken = "BLOCKED_IMMEDIATE_ALERT"
            is_fraud = True
        elif risk_score_pct >= 35.0 or prob_fraud >= threshold:
            risk_level = "MEDIUM"
            action_taken = "FLAGGED_OTP_VERIFICATION"
            is_fraud = True
        else:
            risk_level = "LOW"
            action_taken = "AUTO_APPROVED"
            is_fraud = False
            
        if not audit_flags and risk_level == "LOW":
            audit_flags.append("Standard legitimate transaction pattern")

        return {
            "transaction_id": f"TXN_{uuid.uuid4().hex[:8].upper()}",
            "user_id": tx_dict.get("user_id", "USR_0000"),
            "risk_score": risk_score_pct,
            "risk_level": risk_level,
            "is_fraud": is_fraud,
            "action_taken": action_taken,
            "audit_flags": audit_flags,
            "confidence": round((prob_fraud if is_fraud else (1 - prob_fraud)) * 100, 1),
            "timestamp": tx_time_str
        }

engine = FraudScoringEngine()
