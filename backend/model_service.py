# ============================================================
# ML Model Service
# Financial Fraud Detection System
# ============================================================

import sys
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

# ============================================================
# Project Paths & Sys.path Configuration
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FINAL_MODEL_DIR = PROJECT_ROOT / "final_model"
MODEL_FILE = FINAL_MODEL_DIR / "models" / "ros_logistic_end_to_end_finetuned.pkl"

for directory_path in [str(BACKEND_DIR), str(FINAL_MODEL_DIR)]:
    if directory_path not in sys.path:
        sys.path.insert(0, directory_path)

try:
    from schemas import TransactionRequest
except ImportError:
    from backend.schemas import TransactionRequest



# ============================================================
# Load Trained Model
# ============================================================

try:

    MODEL = joblib.load(
        MODEL_FILE
    )

    print(
        "Final fraud detection model loaded successfully."
    )

except Exception as error:

    MODEL = None

    print(
        "ERROR: Failed to load fraud detection model."
    )

    print(
        error
    )


# ============================================================
# Model Input Columns
# ============================================================

MODEL_FEATURES = [
    "Amount",
    "Account Balance",
    "User Device Recognition",
    "Known Threat Flag",
    "Login Location Match",
    "Spending Pattern Deviation",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "High_Value_Transaction",
    "Low_Balance",
    "Amount_to_Balance_Ratio",
    "Amount_Percent_of_Balance",
    "Spending_Deviation_Amount",
    "Location_Deviation",
    "Unusual_Hour",
    "High_Amount_Unusual_Hour",
    "Location_Deviation_High_Amount",
    "Type_Deposit",
    "Type_Transfer",
    "Type_Withdrawal",
    "Location_California",
    "Location_Florida",
    "Location_New York",
    "Location_Texas",
    "Transaction_Weekday_Friday",
    "Transaction_Weekday_Monday",
    "Transaction_Weekday_Saturday",
    "Transaction_Weekday_Sunday",
    "Transaction_Weekday_Thursday",
    "Transaction_Weekday_Wednesday",
]


# ============================================================
# Model Classification Threshold
# ============================================================

# Calibrated optimal operating threshold: T = 0.42
# Optimization results on holdout set:
# - Boosts fraud recall from 97.0% to 98.5% (catches 197/200 frauds)
# - Halves missed frauds (from 6 to 3)
# - Maximizes F2-score (95.35%) and minimizes expected financial loss
MODEL_THRESHOLD = 0.42


# ============================================================
# Risk Score Thresholds
# ============================================================

LOW_RISK_LIMIT = 30

HIGH_RISK_LIMIT = 70


# ============================================================
# Feature Preparation
# ============================================================

def prepare_features(
    transaction: TransactionRequest
):
    """
    Converts a raw transaction request into the exact
    31-column DataFrame structure expected by the saved model.
    """

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    timestamp = transaction.timestamp

    # --------------------------------------------------------
    # Extract time features
    # --------------------------------------------------------

    transaction_hour = timestamp.hour

    transaction_day = timestamp.day

    transaction_month = timestamp.month

    weekday = pd.Timestamp(timestamp).day_name()

    # --------------------------------------------------------
    # Transaction-level feature engineering
    # --------------------------------------------------------

    amount = transaction.amount

    account_balance = (
        transaction.account_balance
    )

    # Amount / Balance ratio
    if account_balance != 0:

        amount_to_balance_ratio = (
            amount / account_balance
        )

    else:

        amount_to_balance_ratio = (
            amount
        )

    amount_percent_of_balance = (
        amount_to_balance_ratio * 100
    )

    # Spending deviation amount
    spending_deviation_amount = (
        amount
        * transaction.spending_pattern_deviation
    )

    # Location deviation
    location_deviation = (
        1 - transaction.login_location_match
    )

    # Unusual transaction hour
    unusual_hour = int(
        transaction_hour < 6
        or transaction_hour >= 22
    )

    # High-value transaction
    high_value_transaction = int(
        amount > 10000
    )

    # Low-balance transaction
    low_balance = int(
        account_balance < 1000
    )

    # High amount + unusual hour
    high_amount_unusual_hour = (
        high_value_transaction
        * unusual_hour
    )

    # Location deviation + high amount
    location_deviation_high_amount = (
        location_deviation
        * high_value_transaction
    )

    # --------------------------------------------------------
    # Type encoding
    # --------------------------------------------------------

    type_deposit = int(
        transaction.type.lower()
        == "deposit"
    )

    type_transfer = int(
        transaction.type.lower()
        == "transfer"
    )

    type_withdrawal = int(
        transaction.type.lower()
        == "withdrawal"
    )

    # --------------------------------------------------------
    # Location encoding
    # --------------------------------------------------------

    location_lower = (
        transaction.location.lower()
    )

    location_california = int(
        location_lower
        == "california"
    )

    location_florida = int(
        location_lower
        == "florida"
    )

    location_new_york = int(
        location_lower
        == "new york"
    )

    location_texas = int(
        location_lower
        == "texas"
    )

    # --------------------------------------------------------
    # Weekday encoding
    # --------------------------------------------------------

    transaction_weekday_friday = int(
        weekday == "Friday"
    )

    transaction_weekday_monday = int(
        weekday == "Monday"
    )

    transaction_weekday_saturday = int(
        weekday == "Saturday"
    )

    transaction_weekday_sunday = int(
        weekday == "Sunday"
    )

    transaction_weekday_thursday = int(
        weekday == "Thursday"
    )

    transaction_weekday_wednesday = int(
        weekday == "Wednesday"
    )

    # --------------------------------------------------------
    # Build exact model input
    # --------------------------------------------------------

    features = {

        "Amount":
            amount,

        "Account Balance":
            account_balance,

        "User Device Recognition":
            transaction.user_device_recognition,

        "Known Threat Flag":
            transaction.known_threat_flag,

        "Login Location Match":
            transaction.login_location_match,

        "Spending Pattern Deviation":
            transaction.spending_pattern_deviation,

        "Transaction_Hour":
            transaction_hour,

        "Transaction_Day":
            transaction_day,

        "Transaction_Month":
            transaction_month,

        "High_Value_Transaction":
            high_value_transaction,

        "Low_Balance":
            low_balance,

        "Amount_to_Balance_Ratio":
            amount_to_balance_ratio,

        "Amount_Percent_of_Balance":
            amount_percent_of_balance,

        "Spending_Deviation_Amount":
            spending_deviation_amount,

        "Location_Deviation":
            location_deviation,

        "Unusual_Hour":
            unusual_hour,

        "High_Amount_Unusual_Hour":
            high_amount_unusual_hour,

        "Location_Deviation_High_Amount":
            location_deviation_high_amount,

        "Type_Deposit":
            type_deposit,

        "Type_Transfer":
            type_transfer,

        "Type_Withdrawal":
            type_withdrawal,

        "Location_California":
            location_california,

        "Location_Florida":
            location_florida,

        "Location_New York":
            location_new_york,

        "Location_Texas":
            location_texas,

        "Transaction_Weekday_Friday":
            transaction_weekday_friday,

        "Transaction_Weekday_Monday":
            transaction_weekday_monday,

        "Transaction_Weekday_Saturday":
            transaction_weekday_saturday,

        "Transaction_Weekday_Sunday":
            transaction_weekday_sunday,

        "Transaction_Weekday_Thursday":
            transaction_weekday_thursday,

        "Transaction_Weekday_Wednesday":
            transaction_weekday_wednesday,
    }

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        [features],
        columns=MODEL_FEATURES
    )

    return feature_df


# ============================================================
# Risk Level
# ============================================================

def calculate_risk_level(
    risk_score: float
):
    """
    Converts a 0-100 risk score into a risk category.
    """

    if risk_score < LOW_RISK_LIMIT:

        return "Low"

    if risk_score < HIGH_RISK_LIMIT:

        return "Medium"

    return "High"


# ============================================================
# Predict Transaction
# ============================================================

def predict_transaction(
    transaction: TransactionRequest,
    threshold: Optional[float] = None
):
    """
    Performs fraud prediction for one transaction.
    """

    if MODEL is None:
        raise RuntimeError("Fraud detection model is not loaded.")

    active_threshold = MODEL_THRESHOLD if threshold is None else float(threshold)

    # Prepare model input
    features = prepare_features(transaction)

    # Fraud probability
    fraud_probability = float(MODEL.predict_proba(features)[0][1])

    # Prediction using calibrated decision threshold
    prediction_value = int(fraud_probability >= active_threshold)
    prediction = "Suspicious" if prediction_value == 1 else "Normal"

    # Risk scoring (0-100) & risk tier
    risk_score = round(fraud_probability * 100, 2)
    risk_level = calculate_risk_level(risk_score)

    return {
        "transaction_id": transaction.transaction_id,
        "fraud_probability": round(fraud_probability, 4),
        "prediction": prediction,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision_threshold": round(active_threshold, 4),
    }


# ============================================================
# Batch Prediction
# ============================================================

def predict_batch(
    transactions: List[TransactionRequest],
    threshold: Optional[float] = None
):
    """
    Performs batch fraud prediction and aggregates analytics.
    """

    active_threshold = MODEL_THRESHOLD if threshold is None else float(threshold)

    predictions = [
        predict_transaction(tx, threshold=active_threshold)
        for tx in transactions
    ]

    total = len(predictions)
    suspicious_count = sum(1 for p in predictions if p["prediction"] == "Suspicious")
    normal_count = total - suspicious_count
    fraud_rate = round((suspicious_count / total * 100), 2) if total > 0 else 0.0
    avg_risk = round(sum(p["risk_score"] for p in predictions) / total, 2) if total > 0 else 0.0

    return {
        "summary": {
            "total_transactions": total,
            "suspicious_count": suspicious_count,
            "normal_count": normal_count,
            "fraud_rate_percentage": fraud_rate,
            "average_risk_score": avg_risk,
        },
        "decision_threshold": round(active_threshold, 4),
        "predictions": predictions,
    }


# ============================================================
# Model Status
# ============================================================

def is_model_loaded():
    """
    Returns whether the trained ML model is available.
    """

    return MODEL is not None


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    if MODEL is not None:

        print(
            "\nModel service test:"
        )

        print(
            "Model loaded successfully."
        )

        print(
            f"Expected input columns: "
            f"{len(MODEL_FEATURES)}"
        )

        print(
            f"Classification threshold: "
            f"{MODEL_THRESHOLD}"
        )

        print(
            f"Model file: "
            f"{MODEL_FILE}"
        )

    else:

        print(
            "\nModel service test failed."
        )