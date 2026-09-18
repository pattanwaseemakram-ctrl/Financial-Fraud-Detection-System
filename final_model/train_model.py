import os
import sys
import joblib
import pandas as pd

# Add repository root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "final_model"))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from custom_transformers import (
    FeatureSelector,
    SelectiveScaler
)


# Input dataset
INPUT_FILE = os.path.join(
    BASE_DIR,
    "Dataset",
    "processed data",
    "encoded_transactions.csv"
)

# Model output directory
MODEL_DIR = os.path.join(
    BASE_DIR,
    "final_model",
    "models"
)

# Output model file
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "ros_logistic_end_to_end_finetuned.pkl"
)

# Numerical columns that require scaling
NUMERICAL_COLUMNS = [
    "Amount",
    "Account Balance",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month"
]


def main():

    try:
        # Create output directory
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Load encoded dataset
        print(f"Loading dataset from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        print("Dataset shape:", df.shape)

        # Separate features and target
        X = df.drop("Suspicious Activity Flag", axis=1)
        y = df["Suspicious Activity Flag"]

        print("Feature shape:", X.shape)
        print("\nTarget distribution:")
        print(y.value_counts())

        # Hold-out split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print("\nTraining samples:", len(y_train))
        print("Testing samples: ", len(y_test))

        # Complete pipeline
        pipeline = Pipeline([
            (
                "feature_selection",
                FeatureSelector(
                    threshold=0.01,
                    n_estimators=100,
                    random_state=42
                )
            ),
            (
                "scaler",
                SelectiveScaler(
                    numerical_columns=NUMERICAL_COLUMNS
                )
            ),
            (
                "ros",
                RandomOverSampler(
                    sampling_strategy=0.5,
                    random_state=42
                )
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ])

        # Hyperparameter search grid
        parameter_grid = {
            "model__C": [0.1, 1, 10],
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"]
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        print("\nStarting end-to-end model training...")
        print(
            "\nPipeline Architecture:"
            "\nFeature Selection (Random Forest)"
            "\n        |"
            "\n        v"
            "\nSelective Scaling (StandardScaler)"
            "\n        |"
            "\n        v"
            "\nRandom Over-Sampling (ROS)"
            "\n        |"
            "\n        v"
            "\nLogistic Regression Classifier"
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print("\nBest Parameters:", grid_search.best_params_)
        print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_

        # Selected features
        selected_features = (
            best_model
            .named_steps["feature_selection"]
            .selected_features_
        )
        print(f"\nSelected Features ({len(selected_features)}):")
        print(selected_features)

        # Evaluation on hold-out test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"Recall:    {rec:.4f} ({rec*100:.2f}%) - Caught {tp}/{tp+fn} frauds")
        print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"ROC-AUC:   {roc:.4f}")
        print(f"PR-AUC:    {pr:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives:  {tp}")

        # Save model
        joblib.dump(best_model, MODEL_FILE)
        print(f"\nModel successfully saved to: {MODEL_FILE}")

    except Exception as e:
        print(f"Error during model training: {e}")
        raise e


if __name__ == "__main__":
    main()
