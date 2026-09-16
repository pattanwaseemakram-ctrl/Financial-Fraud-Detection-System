import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.linear_model import LogisticRegression

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


# ---------------------------------------------------------
# Add final_model directory to Python path
#
# This allows the script to import custom_transformers.py.
# ---------------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "final_model"
        )
    )
)

from final_model.custom_transformers import (
    FeatureSelector,
    SelectiveScaler
)


# ---------------------------------------------------------
# Input dataset
# ---------------------------------------------------------

INPUT_FILE = (
    "Dataset/processed data/encoded_transactions_improved.csv"
)


# ---------------------------------------------------------
# Output directories
# ---------------------------------------------------------

MODEL_DIR = (
    "final_optimization/models"
)

RESULT_DIR = (
    "final_optimization/results"
)


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "final_ros_logistic_model.pkl"
)

THRESHOLD_FILE = os.path.join(
    RESULT_DIR,
    "final_threshold.json"
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "final_optimization_results.txt"
)

THRESHOLD_RESULT_FILE = os.path.join(
    RESULT_DIR,
    "threshold_analysis.csv"
)


# ---------------------------------------------------------
# Numerical columns requiring scaling
# ---------------------------------------------------------

NUMERICAL_COLUMNS = [
    "Amount",
    "Account Balance",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Amount_to_Balance_Ratio",
    "Amount_Percent_of_Balance",
    "Spending_Deviation_Amount"
]


# ---------------------------------------------------------
# Hyperparameter values
# ---------------------------------------------------------

ROS_RATIOS = [
    0.2,
    0.3,
    0.5,
    0.7,
    1.0
]

C_VALUES = [
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    1.0
]

CLASS_WEIGHTS = [
    None,
    "balanced"
]


def evaluate_predictions(
    y_true,
    y_pred,
    y_prob
):
    """
    Calculate all important classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def main():

    try:

        # ---------------------------------------------------------
        # Create output directories
        # ---------------------------------------------------------

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        os.makedirs(
            RESULT_DIR,
            exist_ok=True
        )

        # ---------------------------------------------------------
        # Load improved encoded dataset
        # ---------------------------------------------------------

        print("Loading improved encoded dataset...")

        df = pd.read_csv(
            INPUT_FILE
        )

        print(
            "Improved encoded dataset loaded successfully."
        )

        print("\nDataset shape:")
        print(
            df.shape
        )

        # ---------------------------------------------------------
        # Validate target column
        # ---------------------------------------------------------

        target_column = (
            "Suspicious Activity Flag"
        )

        if target_column not in df.columns:

            raise ValueError(
                f"Target column '{target_column}' "
                "was not found."
            )

        # ---------------------------------------------------------
        # Separate features and target
        # ---------------------------------------------------------

        X = df.drop(
            target_column,
            axis=1
        )

        y = df[
            target_column
        ]

        print("\nFeature shape:")
        print(
            X.shape
        )

        print("\nTarget distribution:")
        print(
            y.value_counts()
        )

        # ---------------------------------------------------------
        # Create a fresh holdout split
        #
        # IMPORTANT:
        # The holdout is never used during GridSearchCV,
        # threshold selection, or model selection.
        # ---------------------------------------------------------

        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=2026,
            stratify=y
        )

        print("\n")
        print("=" * 60)
        print("FRESH HOLDOUT SPLIT")
        print("=" * 60)

        print("\nDevelopment data shape:")
        print(
            X_dev.shape
        )

        print("\nFresh holdout data shape:")
        print(
            X_holdout.shape
        )

        print("\nDevelopment target distribution:")
        print(
            y_dev.value_counts()
        )

        print("\nFresh holdout target distribution:")
        print(
            y_holdout.value_counts()
        )

        # ---------------------------------------------------------
        # Create base pipeline
        #
        # Feature Selection
        #       ↓
        # Scaling
        #       ↓
        # ROS
        #       ↓
        # Logistic Regression
        #
        # Every transformation is fitted inside each CV fold.
        # ---------------------------------------------------------

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
                    random_state=42
                )
            ),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000
                )
            )
        ])

        # ---------------------------------------------------------
        # Parameter grid
        # ---------------------------------------------------------

        parameter_grid = {
            "ros__sampling_strategy": ROS_RATIOS,
            "model__C": C_VALUES,
            "model__class_weight": CLASS_WEIGHTS
        }

        total_configurations = (
            len(ROS_RATIOS)
            * len(C_VALUES)
            * len(CLASS_WEIGHTS)
        )

        total_fits = (
            total_configurations
            * 5
        )

        print("\n")
        print("=" * 60)
        print("FINAL ROS + LOGISTIC REGRESSION OPTIMIZATION")
        print("=" * 60)

        print(
            f"\nROS ratios: {ROS_RATIOS}"
        )

        print(
            f"C values: {C_VALUES}"
        )

        print(
            f"Class weights: {CLASS_WEIGHTS}"
        )

        print(
            f"\nTotal configurations: "
            f"{total_configurations}"
        )

        print(
            f"Total 5-fold CV fits: "
            f"{total_fits}"
        )

        # ---------------------------------------------------------
        # Stratified cross-validation
        # ---------------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        # ---------------------------------------------------------
        # GridSearchCV
        #
        # F1 is used as the primary model-selection metric.
        # ---------------------------------------------------------

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        print(
            "\nStarting hyperparameter optimization..."
        )

        grid_search.fit(
            X_dev,
            y_dev
        )

        # ---------------------------------------------------------
        # Display best configuration
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("BEST MODEL CONFIGURATION")
        print("=" * 60)

        print("\nBest Parameters:")
        print(
            grid_search.best_params_
        )

        print(
            "\nBest Cross-Validation F1:"
        )

        print(
            f"{grid_search.best_score_:.4f}"
        )

        # ---------------------------------------------------------
        # Get best pipeline
        # ---------------------------------------------------------

        best_model = (
            grid_search.best_estimator_
        )

        # ---------------------------------------------------------
        # Get final selected features
        # ---------------------------------------------------------

        selected_features = (
            best_model
            .named_steps[
                "feature_selection"
            ]
            .selected_features_
        )

        print("\nSelected Features:")
        print(
            selected_features
        )

        print("\nNumber of Selected Features:")
        print(
            len(selected_features)
        )

        # ---------------------------------------------------------
        # Generate out-of-fold probabilities
        #
        # These predictions are used for threshold selection.
        # The fresh holdout is NOT used here.
        # ---------------------------------------------------------

        print(
            "\nGenerating out-of-fold probabilities..."
        )

        oof_probabilities = cross_val_predict(
            best_model,
            X_dev,
            y_dev,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        # ---------------------------------------------------------
        # Calculate OOF ROC-AUC and PR-AUC
        # ---------------------------------------------------------

        oof_roc_auc = roc_auc_score(
            y_dev,
            oof_probabilities
        )

        oof_pr_auc = average_precision_score(
            y_dev,
            oof_probabilities
        )

        print(
            f"\nOOF ROC-AUC: "
            f"{oof_roc_auc:.4f}"
        )

        print(
            f"OOF PR-AUC:  "
            f"{oof_pr_auc:.4f}"
        )

        # ---------------------------------------------------------
        # Search probability thresholds
        #
        # Threshold is chosen using OOF predictions only.
        # ---------------------------------------------------------

        threshold_results = []

        thresholds = np.round(
            np.arange(
                0.10,
                0.91,
                0.01
            ),
            2
        )

        for threshold in thresholds:

            oof_pred = (
                oof_probabilities
                >= threshold
            ).astype(int)

            metrics = evaluate_predictions(
                y_dev,
                oof_pred,
                oof_probabilities
            )

            threshold_results.append({
                "Threshold": threshold,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc"],
                "TN": metrics["tn"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "TP": metrics["tp"]
            })

        threshold_df = pd.DataFrame(
            threshold_results
        )

        # ---------------------------------------------------------
        # Select threshold
        #
        # Primary objective:
        # Maximum OOF F1.
        #
        # Tie-break:
        # Higher precision.
        # ---------------------------------------------------------

        threshold_df = (
            threshold_df
            .sort_values(
                by=[
                    "F1",
                    "Precision",
                    "Recall"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )
            .reset_index(
                drop=True
            )
        )

        best_threshold_row = (
            threshold_df.iloc[0]
        )

        best_threshold = float(
            best_threshold_row[
                "Threshold"
            ]
        )

        threshold_df.to_csv(
            THRESHOLD_RESULT_FILE,
            index=False
        )

        print("\n")
        print("=" * 60)
        print("BEST OOF THRESHOLD")
        print("=" * 60)

        print(
            f"\nThreshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"OOF Accuracy: "
            f"{best_threshold_row['Accuracy']:.4f}"
        )

        print(
            f"OOF Precision: "
            f"{best_threshold_row['Precision']:.4f}"
        )

        print(
            f"OOF Recall: "
            f"{best_threshold_row['Recall']:.4f}"
        )

        print(
            f"OOF F1: "
            f"{best_threshold_row['F1']:.4f}"
        )

        print(
            f"OOF ROC-AUC: "
            f"{best_threshold_row['ROC-AUC']:.4f}"
        )

        print(
            f"OOF PR-AUC: "
            f"{best_threshold_row['PR-AUC']:.4f}"
        )

        print(
            f"OOF False Positives: "
            f"{int(best_threshold_row['FP'])}"
        )

        print(
            f"OOF False Negatives: "
            f"{int(best_threshold_row['FN'])}"
        )

        print(
            f"OOF True Positives: "
            f"{int(best_threshold_row['TP'])}"
        )

        # ---------------------------------------------------------
        # Save threshold
        # ---------------------------------------------------------

        with open(
            THRESHOLD_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "threshold": best_threshold,
                    "model": (
                        "ROS + Logistic Regression"
                    ),
                    "selection_metric": "F1"
                },
                file,
                indent=4
            )

        # ---------------------------------------------------------
        # Generate holdout probabilities
        #
        # The holdout has not been used until this point.
        # ---------------------------------------------------------

        print(
            "\nGenerating fresh holdout predictions..."
        )

        holdout_probabilities = (
            best_model
            .predict_proba(
                X_holdout
            )[:, 1]
        )

        # ---------------------------------------------------------
        # Apply selected threshold
        # ---------------------------------------------------------

        holdout_predictions = (
            holdout_probabilities
            >= best_threshold
        ).astype(int)

        # ---------------------------------------------------------
        # Evaluate fresh holdout
        # ---------------------------------------------------------

        holdout_metrics = evaluate_predictions(
            y_holdout,
            holdout_predictions,
            holdout_probabilities
        )

        # ---------------------------------------------------------
        # Display final holdout results
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("FINAL FRESH HOLDOUT RESULTS")
        print("=" * 60)

        print(
            "\nModel:"
            "\nImproved Features + ROS + "
            "Logistic Regression"
        )

        print(
            f"\nSelected Threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Accuracy:  "
            f"{holdout_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{holdout_metrics['precision']:.4f}"
        )

        print(
            f"Recall:    "
            f"{holdout_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score:  "
            f"{holdout_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:   "
            f"{holdout_metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC:    "
            f"{holdout_metrics['pr_auc']:.4f}"
        )

        # ---------------------------------------------------------
        # Confusion matrix
        # ---------------------------------------------------------

        holdout_cm = confusion_matrix(
            y_holdout,
            holdout_predictions
        )

        print("\nConfusion Matrix:")

        print(
            holdout_cm
        )

        print("\nConfusion Matrix Values:")

        print(
            f"True Negatives:  "
            f"{holdout_metrics['tn']}"
        )

        print(
            f"False Positives: "
            f"{holdout_metrics['fp']}"
        )

        print(
            f"False Negatives: "
            f"{holdout_metrics['fn']}"
        )

        print(
            f"True Positives:  "
            f"{holdout_metrics['tp']}"
        )

        # ---------------------------------------------------------
        # Classification report
        # ---------------------------------------------------------

        print("\nClassification Report:")

        print(
            classification_report(
                y_holdout,
                holdout_predictions,
                zero_division=0
            )
        )

        # ---------------------------------------------------------
        # Refit best model on the complete development dataset
        #
        # The holdout remains untouched.
        # ---------------------------------------------------------

        print(
            "\nRefitting best model on complete "
            "development data..."
        )

        best_model.fit(
            X_dev,
            y_dev
        )

        # ---------------------------------------------------------
        # Save final model
        # ---------------------------------------------------------

        joblib.dump(
            best_model,
            MODEL_FILE
        )

        print(
            "\nFinal optimized model saved to:"
        )

        print(
            MODEL_FILE
        )

        print(
            "\nSelected threshold saved to:"
        )

        print(
            THRESHOLD_FILE
        )

        print(
            "\nThreshold analysis saved to:"
        )

        print(
            THRESHOLD_RESULT_FILE
        )

        # ---------------------------------------------------------
        # Save final text report
        # ---------------------------------------------------------

        with open(
            RESULT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "FINAL ROS + LOGISTIC REGRESSION "
                "OPTIMIZATION REPORT\n"
            )

            file.write(
                "=" * 60
                + "\n\n"
            )

            file.write(
                "Best Parameters:\n"
            )

            file.write(
                str(
                    grid_search.best_params_
                )
                + "\n\n"
            )

            file.write(
                "Best Cross-Validation F1:\n"
            )

            file.write(
                f"{grid_search.best_score_:.4f}\n\n"
            )

            file.write(
                "Selected Features:\n"
            )

            file.write(
                str(
                    selected_features
                )
                + "\n\n"
            )

            file.write(
                f"OOF ROC-AUC: "
                f"{oof_roc_auc:.4f}\n"
            )

            file.write(
                f"OOF PR-AUC: "
                f"{oof_pr_auc:.4f}\n\n"
            )

            file.write(
                f"Selected Threshold: "
                f"{best_threshold:.2f}\n\n"
            )

            file.write(
                "Fresh Holdout Results:\n"
            )

            file.write(
                f"Accuracy:  "
                f"{holdout_metrics['accuracy']:.4f}\n"
            )

            file.write(
                f"Precision: "
                f"{holdout_metrics['precision']:.4f}\n"
            )

            file.write(
                f"Recall:    "
                f"{holdout_metrics['recall']:.4f}\n"
            )

            file.write(
                f"F1 Score:  "
                f"{holdout_metrics['f1']:.4f}\n"
            )

            file.write(
                f"ROC-AUC:   "
                f"{holdout_metrics['roc_auc']:.4f}\n"
            )

            file.write(
                f"PR-AUC:    "
                f"{holdout_metrics['pr_auc']:.4f}\n\n"
            )

            file.write(
                "Confusion Matrix:\n"
            )

            file.write(
                str(
                    holdout_cm
                )
                + "\n\n"
            )

            file.write(
                f"True Negatives:  "
                f"{holdout_metrics['tn']}\n"
            )

            file.write(
                f"False Positives: "
                f"{holdout_metrics['fp']}\n"
            )

            file.write(
                f"False Negatives: "
                f"{holdout_metrics['fn']}\n"
            )

            file.write(
                f"True Positives:  "
                f"{holdout_metrics['tp']}\n\n"
            )

            file.write(
                "Classification Report:\n"
            )

            file.write(
                classification_report(
                    y_holdout,
                    holdout_predictions,
                    zero_division=0
                )
            )

        print(
            "\nFinal optimization completed successfully."
        )

    # Handle missing files
    except FileNotFoundError as e:

        print(
            "\nError: Required file was not found."
        )

        print(e)

    # Handle invalid data
    except ValueError as e:

        print(
            "\nData processing error:"
        )

        print(e)

    # Handle unexpected errors
    except Exception as e:

        print(
            "\nUnexpected error:"
        )

        print(e)


if __name__ == "__main__":
    main()