# 🛡️ Intelligent Fraud Detection System for Financial Transactions

An end-to-end Machine Learning and FastAPI-powered Real-Time Fraud Detection Engine designed for financial institutions and payment gateways to detect, score, and flag suspicious transactions in real time.

---

## 📌 Project Overview
Traditional rule-based fraud detection methods fail to adapt to evolving fraud patterns, generating high false positive rates. This project implements an intelligent system that analyzes transaction behavior in real time, leverages **SMOTE (Synthetic Minority Over-sampling Technique)** to address extreme class imbalance, trains ensemble models (**Random Forest & Logistic Regression**), optimizes decision thresholds to minimize false negatives, and exposes a high-performance REST API with an interactive monitoring dashboard.

---

## 📂 Project Structure

```
Intelligent-Fraud-Detection-System/
│── src/
│   │── feature_engineering.py  # Behavioral & Time Features
│   │── preprocessing.py        # Scaling & SMOTE Oversampling
│   │── train_model.py          # Training & Threshold Optimization
│   │── predict.py              # Real-Time Risk Scoring Engine
│   └── utils.py                # Logging & Helper Utilities
│── app.py                      # FastAPI Web Application & REST API
│── data/
│   └── raw_transactions.csv    # Transaction Dataset
│── models/
│   │── model.joblib            # Trained Random Forest Model
│   │── scaler.joblib           # Feature Scaler
│   └── metrics.json            # Model Evaluation Metrics
│── static/
│   │── css/style.css           # UI Styling
│   └── js/dashboard.js         # Interactive Dashboard Logic
│── templates/
│   └── index.html              # Real-Time Risk Analyzer UI
│── requirements.txt            # Project Dependencies
└── README.md                   # Documentation
```

---

## 🛠️ Step-by-Step Run Instructions

### 1. Activate Environment
```bash
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model
```bash
python -m src.train_model
```

### 4. Launch FastAPI Web Application
```bash
python -m uvicorn app:app --reload --port 8000
```

### 5. Access Dashboard
- **Web Dashboard**: `http://127.0.0.1:8000`
- **Swagger API Documentation**: `http://127.0.0.1:8000/docs`
