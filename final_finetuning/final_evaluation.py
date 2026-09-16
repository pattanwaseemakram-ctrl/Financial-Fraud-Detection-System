import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# ---------------------------------------------------------
# Input dataset
# ---------------------------------------------------------

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "Dataset",
    "processed data",
    "encoded_transactions_improved.csv"
)


# ---------------------------------------------------------
# Final model
# ---------------------------------------------------------

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "final_finetuning",
    "final_clean_model_results",
    "models",
    "final_clean_logistic_model.pkl"
)


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "final_finetuning",
    "final_evaluation_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_evaluation_report.txt"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "final_evaluation_metrics.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_DIR,
    "final_confusion_matrix.csv"
)

ROC_CURVE_FILE = os.path.join(
    OUTPUT_DIR,
    "final_roc_curve.png"
)

PR_CURVE_FILE = os.path.join(
    OUTPUT_DIR,
    "final_precision_recall_curve.png"
)


# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

TARGET_COLUMN = "Suspicious Activity Flag"


# ---------------------------------------------------------
# Features removed during final cleaning
# ---------------------------------------------------------

REMOVED_FEATURES = [
    "Transaction_Month",
    "High_Value_Transaction",
    "Low_Balance",
    "High_Amount_Unusual_Hour",
    "Location_Deviation_High_Amount",
    "Amount_Percent_of_Balance"
]


def main():

    try:

        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------

        print(
            "Loading dataset..."
        )

        df = pd.read_csv(
            INPUT_FILE
        )

        print(
            "Dataset loaded successfully."
        )

        print(
            f"\nOriginal dataset shape: "
            f"{df.shape}"
        )

        # -------------------------------------------------
        # Check target
        # -------------------------------------------------

        if TARGET_COLUMN not in df.columns:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                "was not found."
            )

        # -------------------------------------------------
        # Remove same features used during final training
        # -------------------------------------------------

        existing_removed_features = [
            feature
            for feature in REMOVED_FEATURES
            if feature in df.columns
        ]

        df_clean = df.drop(
            columns=existing_removed_features
        )

        print(
            f"Cleaned dataset shape: "
            f"{df_clean.shape}"
        )

        # -------------------------------------------------
        # Separate features and target
        # -------------------------------------------------

        X = df_clean.drop(
            TARGET_COLUMN,
            axis=1
        )

        y = df_clean[
            TARGET_COLUMN
        ]

        print(
            f"Final feature count: "
            f"{X.shape[1]}"
        )

        # -------------------------------------------------
        # Recreate exact test split
        # used during final training
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print(
            f"Training samples: "
            f"{len(X_train)}"
        )

        print(
            f"Testing samples: "
            f"{len(X_test)}"
        )

        # -------------------------------------------------
        # Load final model
        # -------------------------------------------------

        print("\nLoading final model...")

        model = joblib.load(
            MODEL_FILE
        )

        print(
            "Final model loaded successfully."
        )

        # -------------------------------------------------
        # Generate predictions
        # -------------------------------------------------

        print(
            "\nGenerating predictions..."
        )

        y_pred = model.predict(
            X_test
        )

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        # -------------------------------------------------
        # Calculate metrics
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        pr_auc = average_precision_score(
            y_test,
            y_probability
        )

        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()

        # -------------------------------------------------
        # Display final results
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL EVALUATION RESULTS")
        print("=" * 70)

        print(
            f"\nAccuracy  : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Precision : "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Recall    : "
            f"{recall * 100:.2f}%"
        )

        print(
            f"F1 Score  : "
            f"{f1 * 100:.2f}%"
        )

        print(
            f"ROC-AUC   : "
            f"{roc_auc:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{pr_auc:.4f}"
        )

        print("\nConfusion Matrix:")

        print(
            f"True Negatives  : {tn}"
        )

        print(
            f"False Positives : {fp}"
        )

        print(
            f"False Negatives : {fn}"
        )

        print(
            f"True Positives  : {tp}"
        )

        # -------------------------------------------------
        # Classification report
        # -------------------------------------------------

        classification_report_text = (
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        print("\n")
        print("=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)

        print(
            classification_report_text
        )

        # -------------------------------------------------
        # Save metrics
        # -------------------------------------------------

        metrics_df = pd.DataFrame([
            {
                "Model":
                    "Final Clean Logistic Regression",

                "Test Samples":
                    len(X_test),

                "Features":
                    X.shape[1],

                "Accuracy":
                    accuracy,

                "Precision":
                    precision,

                "Recall":
                    recall,

                "F1":
                    f1,

                "ROC-AUC":
                    roc_auc,

                "PR-AUC":
                    pr_auc,

                "True Negatives":
                    tn,

                "False Positives":
                    fp,

                "False Negatives":
                    fn,

                "True Positives":
                    tp
            }
        ])

        metrics_df.to_csv(
            METRICS_FILE,
            index=False
        )

        # -------------------------------------------------
        # Save confusion matrix
        # -------------------------------------------------

        confusion_df = pd.DataFrame(
            [
                [tn, fp],
                [fn, tp]
            ],
            index=[
                "Actual Normal",
                "Actual Suspicious"
            ],
            columns=[
                "Predicted Normal",
                "Predicted Suspicious"
            ]
        )

        confusion_df.to_csv(
            CONFUSION_MATRIX_FILE
        )

        # -------------------------------------------------
        # ROC curve
        # -------------------------------------------------

        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                y_test,
                y_probability
            )
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"ROC-AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "Final Model ROC Curve"
        )

        plt.legend(
            loc="lower right"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            ROC_CURVE_FILE,
            dpi=300
        )

        plt.close()

        # -------------------------------------------------
        # Precision-Recall curve
        # -------------------------------------------------

        precision_values, recall_values, _ = (
            precision_recall_curve(
                y_test,
                y_probability
            )
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            recall_values,
            precision_values,
            label=f"PR-AUC = {pr_auc:.4f}"
        )

        plt.xlabel(
            "Recall"
        )

        plt.ylabel(
            "Precision"
        )

        plt.title(
            "Final Model Precision-Recall Curve"
        )

        plt.legend(
            loc="upper right"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            PR_CURVE_FILE,
            dpi=300
        )

        plt.close()

        # -------------------------------------------------
        # Save final report
        # -------------------------------------------------

        with open(
            REPORT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "FINAL MODEL EVALUATION REPORT\n"
            )

            file.write(
                "=" * 70
                + "\n\n"
            )

            file.write(
                "Model: Final Clean Logistic Regression\n"
            )

            file.write(
                "Scaling: StandardScaler\n"
            )

            file.write(
                "Balancing: RandomOverSampler\n"
            )

            file.write(
                "ROS Sampling Strategy: 0.5\n"
            )

            file.write(
                "Class Weight: balanced\n"
            )

            file.write(
                "C: 0.0005\n"
            )

            file.write(
                "Solver: liblinear\n\n"
            )

            file.write(
                "Train/Test Split: 80/20\n"
            )

            file.write(
                "Random State: 42\n\n"
            )

            file.write(
                f"Training Samples: "
                f"{len(X_train)}\n"
            )

            file.write(
                f"Testing Samples: "
                f"{len(X_test)}\n"
            )

            file.write(
                f"Final Features: "
                f"{X.shape[1]}\n\n"
            )

            file.write(
                "Removed Features:\n"
            )

            for feature in existing_removed_features:

                file.write(
                    f"- {feature}\n"
                )

            file.write(
                "\nFinal Metrics:\n\n"
            )

            file.write(
                f"Accuracy  : "
                f"{accuracy * 100:.2f}%\n"
            )

            file.write(
                f"Precision : "
                f"{precision * 100:.2f}%\n"
            )

            file.write(
                f"Recall    : "
                f"{recall * 100:.2f}%\n"
            )

            file.write(
                f"F1 Score  : "
                f"{f1 * 100:.2f}%\n"
            )

            file.write(
                f"ROC-AUC   : "
                f"{roc_auc:.4f}\n"
            )

            file.write(
                f"PR-AUC    : "
                f"{pr_auc:.4f}\n\n"
            )

            file.write(
                "Confusion Matrix:\n\n"
            )

            file.write(
                f"True Negatives  : {tn}\n"
            )

            file.write(
                f"False Positives : {fp}\n"
            )

            file.write(
                f"False Negatives : {fn}\n"
            )

            file.write(
                f"True Positives  : {tp}\n\n"
            )

            file.write(
                "Classification Report:\n\n"
            )

            file.write(
                classification_report_text
            )

        # -------------------------------------------------
        # Completion
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL EVALUATION COMPLETED")
        print("=" * 70)

        print("\nFiles saved:")

        print(
            REPORT_FILE
        )

        print(
            METRICS_FILE
        )

        print(
            CONFUSION_MATRIX_FILE
        )

        print(
            ROC_CURVE_FILE
        )

        print(
            PR_CURVE_FILE
        )

    except FileNotFoundError as e:

        print(
            "\nFile not found:"
        )

        print(e)

    except ValueError as e:

        print(
            "\nData error:"
        )

        print(e)

    except Exception as e:

        print(
            "\nUnexpected error:"
        )

        print(e)


if __name__ == "__main__":

    main()