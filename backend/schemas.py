# Request & Response Schemas
# Financial Fraud Detection System

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field



# Transaction Input Schema

class TransactionRequest(BaseModel):
    """
    Input data required for fraud prediction.
    """

    transaction_id: str = Field(
        ...,
        description="Unique transaction identifier",
        example="TX-2026-001"
    )

    timestamp: datetime = Field(
        ...,
        description="Transaction date and time (ISO format)",
        example="2026-09-21T14:30:00"
    )

    sender_account_id: str = Field(
        ...,
        description="Sender account identifier",
        example="ACC-98124"
    )

    receiver_account_id: str = Field(
        ...,
        description="Receiver account identifier",
        example="ACC-12490"
    )

    amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount",
        example=2500.00
    )

    type: str = Field(
        ...,
        description="Transaction type: Deposit, Transfer, or Withdrawal",
        example="Transfer"
    )

    device_id: str = Field(
        ...,
        description="Device identifier",
        example="DEV-4921"
    )

    ip_address: str = Field(
        ...,
        description="IP address used for the transaction",
        example="192.168.1.105"
    )

    location: str = Field(
        ...,
        description="Transaction location (e.g., California, New York, Texas, Florida)",
        example="California"
    )

    account_balance: float = Field(
        ...,
        ge=0,
        description="Account balance before transaction",
        example=12000.50
    )

    user_device_recognition: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether device is recognized (1 = recognized, 0 = unrecognized)",
        example=1
    )

    known_threat_flag: int = Field(
        ...,
        ge=0,
        le=1,
        description="Known threat indicator (1 = flagged blacklist/threat, 0 = clean)",
        example=0
    )

    login_location_match: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether login location matches regular location (1 = match, 0 = mismatch)",
        example=1
    )

    spending_pattern_deviation: float = Field(
        ...,
        description="Deviation from historical spending pattern (z-score or multiple)",
        example=0.15
    )


# Single Prediction Response Schema


class PredictionResponse(BaseModel):
    """
    Response returned by the fraud prediction API for a single transaction.
    """

    transaction_id: str
    fraud_probability: float
    prediction: str
    risk_score: float
    risk_level: str
    decision_threshold: float


# ============================================================
# Batch Prediction Schemas
# ============================================================

class BatchTransactionRequest(BaseModel):
    """
    Request payload containing multiple transactions for batch scoring.
    """

    transactions: List[TransactionRequest]


class BatchSummary(BaseModel):
    """
    High-level aggregate statistics across the evaluated batch.
    """

    total_transactions: int
    suspicious_count: int
    normal_count: int
    fraud_rate_percentage: float
    average_risk_score: float


class BatchPredictionResponse(BaseModel):
    """
    Comprehensive response for batch transaction evaluation.
    """

    summary: BatchSummary
    decision_threshold: float
    predictions: List[PredictionResponse]
