# ============================================================
# Send All 197 Transactions to POST /batch-predict
# Financial Fraud Detection System
# ============================================================

import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_FILE = PROJECT_ROOT / "final_model" / "results" / "request_body_197_transactions.json"

def main():
    print("=" * 70)
    print("SENDING ALL 197 FRAUD TRANSACTIONS TO POST /batch-predict")
    print("=" * 70)

    # 1. Check server health
    print("\n[1/4] Checking API server status...")
    try:
        health_req = urllib.request.urlopen(f"{API_BASE_URL}/health", timeout=3)
        health_data = json.loads(health_req.read().decode())
        print(f"  -> Server Status: {health_data.get('status')} (Database: {health_data.get('database', {}).get('status')})")
    except Exception as e:
        print(f"  -> ERROR: Could not connect to API server at {API_BASE_URL}: {e}")
        print("     Please ensure the server is running with: .\\venv\\Scripts\\uvicorn backend.app:app --port 8000")
        return

    # 2. Authenticate
    print("\n[2/4] Authenticating with OAuth2 (admin:admin123)...")
    auth_data = urllib.parse.urlencode({"username": "admin", "password": "admin123"}).encode()
    auth_req = urllib.request.Request(f"{API_BASE_URL}/auth/token", data=auth_data)
    token = json.loads(urllib.request.urlopen(auth_req).read().decode())["access_token"]
    print("  -> Access token acquired successfully.")

    # 3. Load 197 payload
    print(f"\n[3/4] Reading 197 transactions from {PAYLOAD_FILE.name}...")
    with open(PAYLOAD_FILE, "rb") as f:
        payload_bytes = f.read()

    data = json.loads(payload_bytes.decode())
    tx_count = len(data.get("transactions", []))
    print(f"  -> Total transactions in payload: {tx_count}")

    # 4. Send batch request
    print(f"\n[4/4] Sending POST /batch-predict?threshold=0.42 ...")
    t0 = time.time()
    batch_req = urllib.request.Request(
        f"{API_BASE_URL}/batch-predict?threshold=0.42",
        data=payload_bytes,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    response = json.loads(urllib.request.urlopen(batch_req).read().decode())
    elapsed = time.time() - t0

    summary = response["summary"]
    predictions = response["predictions"]

    print(f"  -> Received HTTP 200 OK in {elapsed:.2f} seconds!")
    print("\n" + "=" * 70)
    print("BATCH PREDICTION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total Transactions Evaluated: {summary['total_transactions']}")
    print(f"  Suspicious Count (Frauds):     {summary['suspicious_count']}")
    print(f"  Normal Count:                  {summary['normal_count']}")
    print(f"  Fraud Detection Rate:          {summary['fraud_rate_percentage']:.1f}%")
    print(f"  Average Risk Score:            {summary['average_risk_score']:.2f} / 100")
    print(f"  Decision Threshold Used:       {response['decision_threshold']}")
    print(f"  Alerts Automatically Synced:   All {summary['suspicious_count']} saved into PostgreSQL 'fraud_alerts' table")

    print("\nFirst 5 Predictions from Response:")
    for p in predictions[:5]:
        print(f"  Transaction {p['transaction_id']}: {p['prediction']} | Prob: {p['fraud_probability']:.4f} | Risk: {p['risk_score']} ({p['risk_level']})")

    print("\nLast 5 Predictions from Response:")
    for p in predictions[-5:]:
        print(f"  Transaction {p['transaction_id']}: {p['prediction']} | Prob: {p['fraud_probability']:.4f} | Risk: {p['risk_score']} ({p['risk_level']})")

if __name__ == "__main__":
    main()
