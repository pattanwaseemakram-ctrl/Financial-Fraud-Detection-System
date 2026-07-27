import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from src.preprocessing import load_and_preprocess_data

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_transactions.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def train():
    print("[INFO] Loading data and applying SMOTE oversampling...")
    data = load_and_preprocess_data(DATA_PATH)
    
    X_train_res = data["X_train_resampled"]
    y_train_res = data["y_train_resampled"]
    X_test_scaled = data["X_test_scaled"]
    y_test = data["y_test"]
    scaler = data["scaler"]
    feature_cols = data["feature_cols"]
    
    # Step 4: Model Selection and Training
    print("[INFO] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    
    probs = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = float(roc_auc_score(y_test, probs))
    print(f"Model ROC-AUC Score: {roc_auc:.4f}")
    
    # Step 5: Threshold Optimization for High Recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = [2 * (p * r) / (p + r + 1e-10) for p, r in zip(precisions, recalls)]
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.35
    
    final_preds = (probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, final_preds).tolist()
    report = classification_report(y_test, final_preds, output_dict=True)
    
    print(f"Optimal Decision Threshold: {optimal_threshold:.4f}")
    print(f"Recall (Sensitivity): {report['1']['recall']*100:.2f}%")
    print(f"Precision: {report['1']['precision']*100:.2f}%")
    
    # Save Model & Scaler Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    metrics = {
        "model_type": "Random Forest Classifier + SMOTE",
        "feature_columns": feature_cols,
        "optimal_threshold": round(optimal_threshold, 4),
        "roc_auc": round(roc_auc, 4),
        "precision_fraud": round(float(report['1']['precision']), 4),
        "recall_fraud": round(float(report['1']['recall']), 4),
        "f1_fraud": round(float(report['1']['f1-score']), 4),
        "accuracy": round(float(report['accuracy']), 4),
        "confusion_matrix": cm
    }
    
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[SUCCESS] Model artifacts successfully saved to {MODELS_DIR}")

if __name__ == "__main__":
    train()
