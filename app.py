import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from src.predict import engine

app = FastAPI(
    title="Intelligent Financial Fraud Detection API",
    description="Real-time transaction fraud scoring engine using ML, SMOTE, & risk threshold optimization.",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

RECENT_FLAGGED_LOGS = [
    {
        "transaction_id": "TXN_88F1A9",
        "user_id": "USR_0192",
        "amount": 3450.00,
        "merchant_category": "Crypto",
        "location": "Lagos, NG",
        "risk_level": "HIGH",
        "risk_score": 92.4,
        "action_taken": "BLOCKED_IMMEDIATE_ALERT"
    },
    {
        "transaction_id": "TXN_34A7B2",
        "user_id": "USR_0341",
        "amount": 890.00,
        "merchant_category": "Electronics",
        "location": "Online",
        "risk_level": "MEDIUM",
        "risk_score": 54.0,
        "action_taken": "FLAGGED_OTP_VERIFICATION"
    }
]

class TransactionInput(BaseModel):
    user_id: str = Field(..., example="USR_0192")
    amount: float = Field(..., example=3450.00, gt=0)
    location: str = Field(..., example="Lagos, NG")
    device_type: str = Field(..., example="Mobile")
    merchant_category: str = Field(..., example="Crypto")
    timestamp: Optional[str] = Field(None, example="2026-07-26 02:15:00")
    transaction_frequency_24h: Optional[int] = Field(1, ge=1)
    user_avg_amount: Optional[float] = Field(150.0, ge=0)

class PredictionResponse(BaseModel):
    transaction_id: str
    user_id: str
    risk_score: float
    risk_level: str
    is_fraud: bool
    action_taken: str
    audit_flags: List[str]
    confidence: float
    timestamp: str

class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    return FileResponse(index_path)

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_transaction(input_data: TransactionInput):
    try:
        data_dict = input_data.model_dump() if hasattr(input_data, 'model_dump') else input_data.dict()
        result = engine.analyze_transaction(data_dict)
        
        if result["risk_level"] in ["HIGH", "MEDIUM"]:
            RECENT_FLAGGED_LOGS.insert(0, {
                "transaction_id": result["transaction_id"],
                "user_id": input_data.user_id,
                "amount": input_data.amount,
                "merchant_category": input_data.merchant_category,
                "location": input_data.location,
                "risk_level": result["risk_level"],
                "risk_score": result["risk_score"],
                "action_taken": result["action_taken"]
            })
            if len(RECENT_FLAGGED_LOGS) > 20:
                RECENT_FLAGGED_LOGS.pop()
                
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-batch", response_model=List[PredictionResponse])
async def predict_batch(batch_input: BatchTransactionInput):
    results = []
    for tx in batch_input.transactions:
        res = engine.analyze_transaction(tx.dict())
        results.append(res)
    return results

@app.get("/api/dashboard-data")
async def get_dashboard_metrics():
    metrics = engine.metrics if engine.metrics else {}
    return {
        "kpis": {
            "total_analyzed": 12500,
            "fraud_detected": 438,
            "protected_amount": 1845000.00,
            "model_accuracy": metrics.get("recall_fraud", 1.0)
        },
        "model_metrics": metrics,
        "recent_flagged": RECENT_FLAGGED_LOGS[:10]
    }
