import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import sqlalchemy as sa

PROJECT_ROOT = Path(r"c:\Users\HP\Documents\GitHub\Financial-Fraud-Detection-System")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "final_model"))

from backend.alert_service import get_engine, initialize_database, alerts_table

CSV_197 = PROJECT_ROOT / "final_model" / "results" / "predicted_197_frauds.csv"

def store_197_frauds():
    if not CSV_197.exists():
        raise FileNotFoundError(f"197 frauds file not found: {CSV_197}")

    df_197 = pd.read_csv(CSV_197)
    print(f"Loaded {len(df_197)} transactions from {CSV_197.name}")

    initialize_database()
    engine = get_engine()

    with engine.begin() as conn:
        # Clear existing alerts to ensure a clean, exact 197-row dataset
        conn.execute(sa.delete(alerts_table))

        # Reset identity/sequence if PostgreSQL
        try:
            conn.execute(sa.text("ALTER SEQUENCE fraud_alerts_id_seq RESTART WITH 1;"))
        except Exception:
            pass

        records = []
        for _, row in df_197.iterrows():
            try:
                dt = datetime.fromisoformat(str(row["Timestamp"]).replace(" ", "T"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)

            records.append({
                "transaction_id": str(row["Transaction ID"]),
                "fraud_probability": float(row["Fraud Probability"]),
                "prediction": "Suspicious",
                "risk_score": float(row["Risk Score"]),
                "risk_level": str(row["Risk Level"]),
                "status": "New",
                "created_at": dt,
                "updated_at": dt,
            })

        conn.execute(sa.insert(alerts_table), records)

    # Verification query
    with engine.connect() as conn:
        count = conn.execute(sa.text("SELECT count(*) FROM fraud_alerts")).scalar()
        summary = conn.execute(sa.text("""
            SELECT risk_level, count(*) 
            FROM fraud_alerts 
            GROUP BY risk_level 
            ORDER BY count(*) DESC
        """)).fetchall()
        sample = conn.execute(sa.text("""
            SELECT id, transaction_id, fraud_probability, risk_score, risk_level, status 
            FROM fraud_alerts 
            ORDER BY id ASC 
            LIMIT 5
        """)).fetchall()

    print("\nSUCCESSFULLY STORED 197 FRAUD ALERTS IN POSTGRESQL!")
    print(f"Total rows in fraud_alerts: {count}")
    print("Breakdown by Risk Level:", summary)
    print("\nFirst 5 Stored Records:")
    for s in sample:
        print(" ", s)

if __name__ == "__main__":
    store_197_frauds()
