# Financial Fraud Detection REST API

import sys
from pathlib import Path
from typing import Optional

# Ensure backend directory is in sys.path for direct or module execution
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import os
import secrets
from fastapi import Depends, FastAPI, HTTPException, Path as FastAPIPath, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

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
        test_connection,
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
        test_connection,
    )


# ============================================================
# FastAPI Application Configuration
# ============================================================

tags_metadata = [
    {
        "name": "Authentication & Security",
        "description": "OAuth2 password flow (`fraudGuardAuth`) for issuing role-based Bearer access tokens.",
    },
    {
        "name": "Fraud Prediction Engine",
        "description": "Real-time and batch machine learning inference powered by calibrated decision thresholding ($T = 0.42$).",
    },
    {
        "name": "Fraud Alert Management",
        "description": "Lifecycle triage and management for flagged suspicious transactions in PostgreSQL (with automatic SQLite fallback).",
    },
    {
        "name": "System Health & Monitoring",
        "description": "Public health checks and service discovery for DevOps and uptime monitoring.",
    },
]

app = FastAPI(
    title="Financial Fraud Detection API",
    description=(
        "Enterprise real-time and batch fraud prediction API powered by an "
        "end-to-end trained ML pipeline (ROS + Logistic Regression) with "
        "cost-calibrated decision thresholding and fraud alert management."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
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
# API Security Configuration (Role-Based OAuth2 - fraudGuardAuth)
# ============================================================

# Domain-specific OAuth2 Password Bearer scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scheme_name="bearerAuth",
    description="OAuth2 Password Flow. Enter username 'admin' and password 'admin123'. Leave client_id/secret empty.",
)

# User credentials database
USER_DATABASE = {
    "admin": {
        "password": os.getenv("FRAUD_ADMIN_PASSWORD", "admin123"),
        "role": "Chief Risk Officer",
    },
    "analyst": {
        "password": os.getenv("FRAUD_ANALYST_PASSWORD", "analyst123"),
        "role": "Fraud Investigation Analyst",
    },
}

# Static master token for automated test systems
STATIC_MASTER_TOKEN = os.getenv("FRAUD_API_TOKEN", "fraud-secret-bearer-token-2026")

# In-memory session tokens cache
ACTIVE_TOKENS = {
    STATIC_MASTER_TOKEN: {
        "username": "admin",
        "role": "Chief Risk Officer",
    },
}


def verify_bearer_token(token: str = Security(oauth2_scheme)) -> dict:
    """
    Validates OAuth2 Bearer token from the Authorization header.
    Renders 'fraudGuardAuth (OAuth2, password)' in Swagger UI (/docs).
    """
    if not token or (token not in ACTIVE_TOKENS and token != STATIC_MASTER_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or expired Bearer token. Log in via 'Authorize' (lock symbol) in Swagger UI.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return ACTIVE_TOKENS.get(token) or ACTIVE_TOKENS[STATIC_MASTER_TOKEN]





# ============================================================
# OAuth2 Token Endpoint
# ============================================================

@app.post(
    "/auth/token",
    summary="Obtain OAuth2 Bearer Access Token",
    tags=["Authentication & Security"],
)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticates user credentials and issues a Bearer access token.
    Used by Swagger UI ('fraudGuardAuth') when clicking Authorize.

    Available accounts:
    - `admin` / `admin123` (Chief Risk Officer)
    - `analyst` / `analyst123` (Fraud Analyst)
    """
    username = form_data.username
    password = form_data.password

    user = USER_DATABASE.get(username)
    if not user or user["password"] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password. Available users: 'admin' (pass: admin123), 'analyst' (pass: analyst123)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Issue unique session token
    access_token = f"fraud_token_{secrets.token_hex(16)}"
    ACTIVE_TOKENS[access_token] = {
        "username": username,
        "role": user["role"],
    }

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_role": user["role"],
    }


@app.post("/token", include_in_schema=False)
def legacy_token_alias(form_data: OAuth2PasswordRequestForm = Depends()):
    """Backward compatibility alias for /token."""
    return login_for_access_token(form_data)


# ============================================================
# Root Endpoint
# ============================================================

@app.get(
    "/",
    summary="API Root & Service Discovery",
    tags=["System Health & Monitoring"],
)
def root():
    return {
        "service": "Financial Fraud Detection API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "default_threshold": MODEL_THRESHOLD,
        "authentication": {
            "type": "Role-Based OAuth2 (fraudGuardAuth)",
            "token_url": "/auth/token",
            "demo_credentials": {
                "admin": "admin123 (Chief Risk Officer)",
                "analyst": "analyst123 (Fraud Analyst)",
            },
            "instructions": "In Swagger UI (/docs), click 'Authorize' (lock symbol), enter username and password, and click Authorize.",
        },
    }


# ============================================================
# Health Check Endpoint
# ============================================================

@app.get(
    "/health",
    summary="Service health & model readiness",
    tags=["System Health & Monitoring"],
)
def health_check():
    model_ok = is_model_loaded()
    db_status = test_connection()

    return {
        "status": "healthy" if model_ok and db_status.get("status") == "connected" else "degraded",
        "model_loaded": model_ok,
        "database": db_status,
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
    tags=["Fraud Prediction Engine"],
    dependencies=[Security(verify_bearer_token)],
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
    tags=["Fraud Prediction Engine"],
    dependencies=[Security(verify_bearer_token)],
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
    tags=["Fraud Alert Management"],
    dependencies=[Security(verify_bearer_token)],
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
    tags=["Fraud Alert Management"],
    dependencies=[Security(verify_bearer_token)],
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
    tags=["Fraud Alert Management"],
    dependencies=[Security(verify_bearer_token)],
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
    tags=["Fraud Alert Management"],
    dependencies=[Security(verify_bearer_token)],
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
    tags=["Fraud Alert Management"],
    dependencies=[Security(verify_bearer_token)],
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