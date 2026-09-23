# Financial Fraud Detection System

An enterprise-grade Machine Learning system designed to detect fraudulent financial transactions in real time while maintaining high precision, high recall, and near-zero false alarms.

---

## Performance Benchmark

Evaluated on an untouched hold-out test set of **2,000 transactions** (1,800 legitimate, 200 fraudulent):

| Metric | Baseline ($T = 0.50$) | Optimized ($T = 0.42$) | Performance Impact |
| :--- | :---: | :---: | :--- |
| **Recall (Fraud Caught)** | 97.00% (194/200) | **98.50% (197/200)** | 🟢 **Catches 3 additional frauds** |
| **Missed Frauds (FN)** | 6 missed | **3 missed** | 🟢 **Missed fraud rate cut by 50%** |
| **Precision** | **87.39%** | 84.55% | Only 36 false alarms out of 1,800 normal |
| **Accuracy** | **98.30%** | 98.05% | Near-zero false alarm rate (2.0%) |
| **$F_1$-Score** | 91.94% | 90.99% | Robust harmonic balance |
| **$F_2$-Score (Fraud-Weighted)** | 94.91% | **95.35%** | 🟢 **Optimal recall-weighted fraud metric** |
| **ROC-AUC** | **0.9967** | **0.9967** | Outstanding discriminative ability |
| **PR-AUC** | **0.9712** | **0.9712** | Superior performance on imbalanced class |

### Confusion Matrix (Optimized Operating Threshold = 0.42)
```
                     Predicted Normal (0)   Predicted Suspicious (1)
Actual Normal (0)            1,764                      36          (False Alarms)
Actual Suspicious (1)            3                     197          (Frauds Caught)
                             (Missed)
```

---

## System Architecture

```
Financial-Fraud-Detection-System/
│
├── Dataset/
│   ├── raw dataset/
│   │   └── financial_transactions.csv            # Raw transaction records (10,000 rows)
│   ├── processed data/
│   │   ├── cleaned_transactions.csv              # Preprocessed & validated transactions
│   │   ├── feature_engineered_transactions.csv   # Domain-engineered risk features
│   │   ├── encoded_transactions.csv              # One-hot encoded dataset
│   │   └── encoded_transactions_improved.csv     # Extended feature dataset
│   └── split/
│       ├── X_train.csv / X_test.csv              # Scaled features (80% train / 20% test)
│       ├── y_train.csv / y_test.csv              # Ground-truth targets
│       ├── X_train_selected.csv                  # Selected top 11 features (train)
│       ├── X_test_selected.csv                   # Selected top 11 features (test)
│       └── feature_importance.png                # Random Forest feature importance chart
│
├── pipeline/
│   ├── data_preprocessing.py                     # Data hygiene, missing value & type checks
│   ├── feature_engineering.py                    # Behavioral ratios, time & interaction features
│   ├── encoding.py                               # Categorical one-hot encoding
│   ├── feature_scaling.py                        # StandardScaler & stratified 80/20 train/test split
│   ├── feature_selection.py                      # Random Forest feature importance ranking
│   ├── eda.PY                                    # Exploratory data analysis
│   ├── correlation_matrix.py                     # Feature correlation computation
│   ├── outlier_detection.py                      # Outlier detection analysis
│   └── class_imbalance_check.py                  # Imbalance analysis & resampling helpers
│
├── final_model/
│   ├── custom_transformers.py                    # FeatureSelector & SelectiveScaler transformers
│   ├── train_model.py                            # End-to-end model training (ROS + Logistic Regression)
│   ├── threshold_optimization.py                 # Multi-criterion threshold tuning & financial cost optimization
│   ├── final_prediction.py                       # Batch prediction engine with risk tiers & probabilities
│   ├── models/
│   │   └── ros_logistic_end_to_end_finetuned.pkl # Production trained pipeline model
│   ├── evaluation/
│   │   ├── final_evaluation.py                   # Evaluation suite on untouched test data
│   │   └── results/
│   │       ├── final_evaluation_report.txt       # Text report of metrics & confusion matrix
│   │       ├── confusion_matrix.png              # High-resolution confusion matrix heatmap
│   │       ├── threshold_optimization_report.txt # Multi-strategy threshold report & cost benchmark
│   │       ├── threshold_sweep_results.csv       # Sweep data across candidate cutoffs (0.05 to 0.95)
│   │       ├── threshold_metrics_curve.png       # Precision, Recall, F1, F2 vs threshold plot
│   │       └── threshold_cost_curve.png          # Total financial loss vs threshold curve
│   ├── visualization/
│   │   ├── final_curves.py                       # Script to generate ROC and PR curves
│   │   └── results/
│   │       ├── roc_curve.png                     # ROC curve (AUC = 0.9967)
│   │       └── precision_recall_curve.png        # Precision-Recall curve (PR-AUC = 0.9712)
│   └── results/
│       └── final_fraud_predictions.csv           # Scored transactions with fraud probabilities & risk tiers
│
├── backend/
│   ├── app.py                                    # FastAPI REST API application & endpoints
│   ├── model_service.py                          # Feature preparation & calibrated model scoring service
│   ├── schemas.py                                # Pydantic request & response validation models
│   └── alert_service.py                          # SQLite database service for fraud alert management & triage
│
├── requirements.txt                              # Pinned Python package dependencies
├── .gitignore                                    # Version control exclusion rules
└── README.md                                     # Project documentation
```

---

## Machine Learning Pipeline

```
Raw Data (10,000 Transactions)
       |
       v
1. Preprocessing (pipeline/data_preprocessing.py)
   - Null value validation & duplicate elimination
   - Timestamp parsing & type conversion
       |
       v
2. Feature Engineering (pipeline/feature_engineering.py)
   - Spending_Deviation_Amount = Amount * Spending_Pattern_Deviation
   - Location_Deviation = 1 - Login_Location_Match
   - Amount_to_Balance_Ratio & Amount_Percent_of_Balance
   - Unusual_Hour flag (midnight to 5 AM / late evening)
       |
       v
3. Categorical Encoding (pipeline/encoding.py)
   - One-hot encoding for Type, Location, and Transaction_Weekday
       |
       v
4. Stratified Split & Scaling (pipeline/feature_scaling.py)
   - 80% Train (8,000) / 20% Test (2,000) stratified hold-out
   - StandardScaler fitted strictly on training data (Zero data leakage)
       |
       v
5. Feature Selection (pipeline/feature_selection.py)
   - Random Forest importance ranking selects the top 11 most predictive features
       |
       v
6. End-to-End Modeling (final_model/train_model.py)
   - Random Over-Sampling (ROS) balances the 10% minority fraud class
   - Logistic Regression with 5-fold Stratified Cross-Validation
   - Exported as a complete deployable pipeline
       |
       v
7. Evaluation & Predictions (final_model/)
   - final_evaluation.py -> Generates metrics & confusion matrix
   - final_curves.py     -> Generates ROC & Precision-Recall curves
   - final_prediction.py -> Assigns probabilities and risk categories (Low, Medium, High)
```

---

## Key Features Used by the Model

The Random Forest feature selection identified the top 11 predictive features:

1. **`Spending_Deviation_Amount`**: High transaction value coinciding with abnormal spending behavior.
2. **`User Device Recognition`**: Whether the device was unrecognized (strong fraud signal).
3. **`Spending Pattern Deviation`**: Behavioral anomaly flag.
4. **`Login Location Match`**: Login location mismatch indicator.
5. **`Known Threat Flag`**: Previous malicious/blacklist flag.
6. **`Location_Deviation`**: Inverted login location flag.
7. **`Amount_Percent_of_Balance`**: Ratio of amount to available balance.
8. **`Amount`**: Transaction volume.
9. **`Amount_to_Balance_Ratio`**: Direct drain ratio.
10. **`Account Balance`**: Available customer balance.
11. **`Transaction_Hour`**: Time of day transaction was initiated.

---

## How to Run

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Processing Pipeline (Optional - data is pre-generated)
```bash
python pipeline/data_preprocessing.py
python pipeline/feature_engineering.py
python pipeline/encoding.py
python pipeline/feature_scaling.py
python pipeline/feature_selection.py
```

### 3. Train the Model
Train the end-to-end model pipeline with cross-validation and export the model artifact:
```bash
python final_model/train_model.py
```

### 4. Evaluate on Test Set
Evaluate the trained model on the 2,000 hold-out test samples:
```bash
python final_model/evaluation/final_evaluation.py
```
*Outputs: `final_model/evaluation/results/final_evaluation_report.txt` and `confusion_matrix.png`*

### 5. Generate ROC & Precision-Recall Curves
```bash
python final_model/visualization/final_curves.py
```
*Outputs: `final_model/visualization/results/roc_curve.png` and `precision_recall_curve.png`*

### 6. Perform Decision Threshold & Cost Optimization
Tune the decision threshold across criteria ($F_1$, $F_2$, Youden's $J$, and expected financial cost):
```bash
python final_model/threshold_optimization.py
```
*Outputs: `threshold_optimization_report.txt`, `threshold_metrics_curve.png`, `threshold_cost_curve.png`, and `threshold_sweep_results.csv`*

### 7. Run Batch Predictions & Risk Scoring
Generate predictions and risk levels across transactions:
```bash
python final_model/final_prediction.py
```
*Outputs: `final_model/results/final_fraud_predictions.csv`*

### 8. Run Real-Time Fraud Detection REST API (FastAPI)
Launch the production REST service:
```bash
uvicorn backend.app:app --reload --port 8000
```
- **Interactive Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Health Check & Model Status**: `http://localhost:8000/health`
- **Single Transaction Scoring**: `POST /predict` (with optional `?threshold=0.42`)
- **Multi-Transaction Batch Scoring**: `POST /batch-predict`
- **Fraud Alerts Management**: `GET /alerts`, `GET /alerts-summary`, `PUT /alerts/{id}/status`

---

## Technologies Used

- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `imbalanced-learn`, `xgboost`
- **Visualization:** `matplotlib`, `seaborn`
- **API & Serving:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Serialization:** `joblib`
