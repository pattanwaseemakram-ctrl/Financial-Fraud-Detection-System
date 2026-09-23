# ============================================================
# Financial Fraud Detection REST API
# ============================================================

import sys
from pathlib import Path
from typing import Optional

# Ensure backend directory is in sys.path for direct or module execution
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, HTTPException, Query, Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware

try:
    from schemas import (
        TransactionRequest,
        PredictionResponse,
        BatchTransactionRequest,
        BatchPredictionResponse,
    )

    from model_service import (
        predict_transaction,
        predict_batch,
        is_model_loaded,
        MODEL_THRESHOLD,
    )

    from alert_service import (
        initialize_database,
        create_alert,
        get_all_alerts,
        get_alert,
        update_alert_status,
        delete_alert,
        get_alert_summary,
    )

except ImportError:
    from backend.schemas import (
        TransactionRequest,
        PredictionResponse,
        BatchTransactionRequest,
        BatchPredictionResponse,
    )

    from backend.model_service import (
        predict_transaction,
        predict_batch,
        is_model_loaded,
        MODEL_THRESHOLD,
    )

    from backend.alert_service import (
        initialize_database,
        create_alert,
        get_all_alerts,
        get_alert,
        update_alert_status,
        delete_alert,
        get_alert_summary,
    )


# ============================================================
# FastAPI Application Configuration
# ============================================================

app = FastAPI(
    title="Financial Fraud Detection API",
    description=(
        "Enterprise real-time and batch fraud prediction API powered by an "
        "end-to-end trained ML pipeline (ROS + Logistic Regression) with "
        "cost-calibrated decision thresholding and fraud alert management."
    ),
    version="1.0.0",
)


# ============================================================
# CORS Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Initialize Alert Database
# ============================================================

try:
    initialize_database()
except Exception as error:
    print(f"Warning: Alert database initialization failed: {error}")


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Financial Fraud Detection API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "default_threshold": MODEL_THRESHOLD,
    }


# ============================================================
# Health Check Endpoint
# ============================================================

@app.get("/health")
def health_check():
    model_ok = is_model_loaded()

    return {
        "status": "healthy" if model_ok else "degraded",
        "model_loaded": model_ok,
        "calibrated_threshold": MODEL_THRESHOLD,
        "algorithm": "ROS + Logistic Regression (Pipeline)",
    }


# ============================================================
# Real-Time Single Transaction Scoring Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Real-time transaction fraud scoring",
)
def predict(
    transaction: TransactionRequest,
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Optional custom cutoff threshold (defaults to calibrated 0.42)",
    ),
):
    """
    Receives a single financial transaction and returns:

    - Fraud probability
    - Prediction ('Normal' vs 'Suspicious')
    - Risk score
    - Risk level
    - Active decision threshold

    If the transaction is classified as Suspicious,
    an alert is automatically stored in the database.
    """

    try:
        # --------------------------------------------
        # Run ML prediction
        # --------------------------------------------
        result = predict_transaction(
            transaction,
            threshold=threshold
        )

        # --------------------------------------------
        # Extract prediction values
        # --------------------------------------------
        transaction_id = result["transaction_id"]
        fraud_probability = result["fraud_probability"]
        prediction = result["prediction"]
        risk_score = result["risk_score"]
        risk_level = result["risk_level"]

        # --------------------------------------------
        # Create alert for suspicious transactions
        # --------------------------------------------
        alert_result = create_alert(
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            prediction=prediction,
            risk_score=risk_score,
            risk_level=risk_level,
        )

        # --------------------------------------------
        # Return original prediction response
        # --------------------------------------------
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(error)}"
        )


# ============================================================
# Batch Transactions Scoring Endpoint
# ============================================================

@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    summary="Batch transaction scoring & risk aggregation",
)
def batch_predict(
    payload: BatchTransactionRequest,
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Optional custom cutoff threshold (defaults to calibrated 0.42)",
    ),
):
    """
    Receives multiple transactions and returns:

    - Aggregated batch summary
    - Individual predictions
    - Fraud probability
    - Risk score
    - Risk level

    Suspicious transactions are automatically stored as alerts.
    """

    try:
        # --------------------------------------------
        # Run batch prediction
        # --------------------------------------------
        result = predict_batch(
            payload.transactions,
            threshold=threshold
        )

        # --------------------------------------------
        # Save suspicious transactions as alerts
        # --------------------------------------------
        predictions = result.get("predictions", [])

        for prediction_result in predictions:

            create_alert(
                transaction_id=prediction_result["transaction_id"],
                fraud_probability=prediction_result["fraud_probability"],
                prediction=prediction_result["prediction"],
                risk_score=prediction_result["risk_score"],
                risk_level=prediction_result["risk_level"],
            )

        # --------------------------------------------
        # Return batch result
        # --------------------------------------------
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Batch inference error: {str(error)}"
        )


# ============================================================
# Get All Fraud Alerts
# ============================================================

@app.get(
    "/alerts",
    summary="Get fraud alerts",
)
def get_alerts(
    status: Optional[str] = Query(
        None,
        description="Optional alert status filter: New, Under Review, Resolved",
    )
):
    """
    Returns all stored fraud alerts.

    Optional status filtering is supported.
    """

    try:
        alerts = get_all_alerts(status=status)

        return {
            "total_alerts": len(alerts),
            "alerts": alerts,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Alert retrieval error: {str(error)}"
        )


# ============================================================
# Get Alert by ID
# ============================================================

@app.get(
    "/alerts/{alert_id}",
    summary="Get a specific fraud alert",
)
def get_single_alert(
    alert_id: int = FastAPIPath(
        ...,
        ge=1,
        description="Alert ID",
    )
):
    """
    Returns a single fraud alert using its ID.
    """

    try:
        alert = get_alert(alert_id)

        if alert is None:
            raise HTTPException(
                status_code=404,
                detail="Alert not found"
            )

        return alert

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Alert retrieval error: {str(error)}"
        )


# ============================================================
# Update Alert Status
# ============================================================

@app.put(
    "/alerts/{alert_id}/status",
    summary="Update fraud alert status",
)
def change_alert_status(
    alert_id: int = FastAPIPath(
        ...,
        ge=1,
        description="Alert ID",
    ),
    status: str = Query(
        ...,
        description="New status: New, Under Review, Resolved",
    ),
):
    """
    Updates the status of an existing alert.
    """

    allowed_statuses = {
        "New",
        "Under Review",
        "Resolved",
    }

    if status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid status. "
                "Allowed values: New, Under Review, Resolved"
            ),
        )

    try:
        result = update_alert_status(
            alert_id=alert_id,
            status=status,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Alert status update error: {str(error)}"
        )


# ============================================================
# Delete Alert
# ============================================================

@app.delete(
    "/alerts/{alert_id}",
    summary="Delete a fraud alert",
)
def remove_alert(
    alert_id: int = FastAPIPath(
        ...,
        ge=1,
        description="Alert ID",
    )
):
    """
    Deletes an existing alert.
    """

    try:
        result = delete_alert(alert_id)

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Alert deletion error: {str(error)}"
        )


# ============================================================
# Alert Summary
# ============================================================

@app.get(
    "/alerts-summary",
    summary="Get fraud alert summary",
)
def alerts_summary():
    """
    Returns alert counts by status.
    """

    try:
        return get_alert_summary()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Alert summary error: {str(error)}"
        )


# ============================================================
# Run Application Directly
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )