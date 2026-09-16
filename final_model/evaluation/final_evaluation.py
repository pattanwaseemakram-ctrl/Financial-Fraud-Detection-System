import os
import sys
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
    ConfusionMatrixDisplay
)


# ---------------------------------------------------------
# Add final_model directory to Python path
#
# This allows the script to import custom_transformers.py
# from the parent final_model folder.
# ---------------------------------------------------------

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# Import custom transformers
from custom_transformers import (
    FeatureSelector,
    SelectiveScaler
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

# Input dataset
INPUT_FILE = os.path.join(
    BASE_DIR,
    "Dataset",
    "processed data",
    "encoded_transactions.csv"
)

# Trained final model
MODEL_FILE = os.path.join(
    BASE_DIR,
    "final_finetuning",
    "models",
    "ros_logistic_end_to_end_finetuned.pkl"
)

# Evaluation output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_model",
    "evaluation",
    "results"
)

# Evaluation report
REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_evaluation_report.txt"
)

# Confusion matrix image
CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


def main():

    try:

        # ---------------------------------------------------------
        # Create output directory
        # ---------------------------------------------------------

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        # ---------------------------------------------------------
        # Load dataset
        # ---------------------------------------------------------

        print("Loading dataset...")

        df = pd.read_csv(
            INPUT_FILE
        )

        print("Dataset loaded successfully.")

        print("\nDataset shape:")
        print(df.shape)

        # ---------------------------------------------------------
        # Separate features and target
        # ---------------------------------------------------------

        target_column = "Suspicious Activity Flag"

        if target_column not in df.columns:

            raise ValueError(
                f"Target column '{target_column}' was not found."
            )

        X = df.drop(
            target_column,
            axis=1
        )

        y = df[
            target_column
        ]

        # ---------------------------------------------------------
        # Recreate the same train/test split used during
        # final model training
        # ---------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print("\nTest data shape:")
        print(X_test.shape)

        print("\nTest target distribution:")
        print(y_test.value_counts())

        # ---------------------------------------------------------
        # Load final trained model
        # ---------------------------------------------------------

        print("\nLoading final model...")

        model = joblib.load(
            MODEL_FILE
        )

        print("Final model loaded successfully.")

        # ---------------------------------------------------------
        # Generate predictions
        # ---------------------------------------------------------

        print("\nGenerating final evaluation predictions...")

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # ---------------------------------------------------------
        # Calculate evaluation metrics
        # ---------------------------------------------------------

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
            y_prob
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        # ---------------------------------------------------------
        # Confusion matrix
        # ---------------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        tn, fp, fn, tp = cm.ravel()

        # ---------------------------------------------------------
        # Classification report
        # ---------------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        # ---------------------------------------------------------
        # Display final evaluation results
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("FINAL MODEL EVALUATION")
        print("=" * 60)

        print(
            "\nModel:"
            "\nROS + Logistic Regression"
        )

        print("\nTest Metrics:")

        print(
            f"Accuracy:  {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"F1 Score:  {f1:.4f}"
        )

        print(
            f"ROC-AUC:   {roc_auc:.4f}"
        )

        print(
            f"PR-AUC:    {pr_auc:.4f}"
        )

        print("\nConfusion Matrix:")

        print(cm)

        print("\nConfusion Matrix Values:")

        print(
            f"True Negatives:  {tn}"
        )

        print(
            f"False Positives: {fp}"
        )

        print(
            f"False Negatives: {fn}"
        )

        print(
            f"True Positives:  {tp}"
        )

        print("\nClassification Report:")

        print(report)

        # ---------------------------------------------------------
        # Save evaluation report
        # ---------------------------------------------------------

        with open(
            REPORT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "FINAL MODEL EVALUATION REPORT\n"
            )

            file.write(
                "=" * 60
                + "\n"
            )

            file.write(
                "Model: ROS + Logistic Regression\n\n"
            )

            file.write(
                f"Test Dataset Size: {len(y_test)}\n\n"
            )

            file.write(
                "Metrics:\n"
            )

            file.write(
                f"Accuracy:  {accuracy:.4f}\n"
            )

            file.write(
                f"Precision: {precision:.4f}\n"
            )

            file.write(
                f"Recall:    {recall:.4f}\n"
            )

            file.write(
                f"F1 Score:  {f1:.4f}\n"
            )

            file.write(
                f"ROC-AUC:   {roc_auc:.4f}\n"
            )

            file.write(
                f"PR-AUC:    {pr_auc:.4f}\n\n"
            )

            file.write(
                "Confusion Matrix:\n"
            )

            file.write(
                str(cm)
                + "\n\n"
            )

            file.write(
                f"True Negatives:  {tn}\n"
            )

            file.write(
                f"False Positives: {fp}\n"
            )

            file.write(
                f"False Negatives: {fn}\n"
            )

            file.write(
                f"True Positives:  {tp}\n\n"
            )

            file.write(
                "Classification Report:\n"
            )

            file.write(
                report
            )

        # ---------------------------------------------------------
        # Create confusion matrix visualization
        # ---------------------------------------------------------

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Normal",
                "Suspicious"
            ]
        )

        display.plot()

        plt.title(
            "Confusion Matrix - ROS + Logistic Regression"
        )

        plt.tight_layout()

        plt.savefig(
            CONFUSION_MATRIX_FILE,
            dpi=300
        )

        plt.close()

        # ---------------------------------------------------------
        # Display saved files
        # ---------------------------------------------------------

        print("\nEvaluation report saved to:")
        print(REPORT_FILE)

        print("\nConfusion matrix saved to:")
        print(CONFUSION_MATRIX_FILE)

        print(
            "\nFinal model evaluation completed successfully."
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