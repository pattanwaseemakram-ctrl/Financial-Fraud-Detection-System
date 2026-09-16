import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
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

# Final trained model
MODEL_FILE = os.path.join(
    BASE_DIR,
    "final_finetuning",
    "models",
    "ros_logistic_end_to_end_finetuned.pkl"
)

# Visualization output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_model",
    "visualization",
    "results"
)

# ROC curve output
ROC_FILE = os.path.join(
    OUTPUT_DIR,
    "roc_curve.png"
)

# Precision-Recall curve output
PR_FILE = os.path.join(
    OUTPUT_DIR,
    "precision_recall_curve.png"
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

        # ---------------------------------------------------------
        # Load final trained model
        # ---------------------------------------------------------

        print("\nLoading final model...")

        model = joblib.load(
            MODEL_FILE
        )

        print("Final model loaded successfully.")

        # ---------------------------------------------------------
        # Generate probability predictions
        #
        # Class 1 represents suspicious transactions.
        # ---------------------------------------------------------

        print("\nGenerating prediction probabilities...")

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # ---------------------------------------------------------
        # Calculate ROC-AUC
        # ---------------------------------------------------------

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        # ---------------------------------------------------------
        # Generate ROC curve values
        # ---------------------------------------------------------

        false_positive_rate, true_positive_rate, roc_thresholds = (
            roc_curve(
                y_test,
                y_prob
            )
        )

        # ---------------------------------------------------------
        # Create ROC curve
        # ---------------------------------------------------------

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
            linestyle="--",
            label="Random Classifier"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve - ROS + Logistic Regression"
        )

        plt.legend(
            loc="lower right"
        )

        plt.tight_layout()

        plt.savefig(
            ROC_FILE,
            dpi=300
        )

        plt.close()

        # ---------------------------------------------------------
        # Calculate Precision-Recall values
        # ---------------------------------------------------------

        precision, recall, pr_thresholds = (
            precision_recall_curve(
                y_test,
                y_prob
            )
        )

        # ---------------------------------------------------------
        # Calculate PR-AUC
        # ---------------------------------------------------------

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        # ---------------------------------------------------------
        # Create Precision-Recall curve
        # ---------------------------------------------------------

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            recall,
            precision,
            label=f"PR-AUC = {pr_auc:.4f}"
        )

        plt.xlabel(
            "Recall"
        )

        plt.ylabel(
            "Precision"
        )

        plt.title(
            "Precision-Recall Curve - ROS + Logistic Regression"
        )

        plt.legend(
            loc="upper right"
        )

        plt.tight_layout()

        plt.savefig(
            PR_FILE,
            dpi=300
        )

        plt.close()

        # ---------------------------------------------------------
        # Display results
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("FINAL CURVE RESULTS")
        print("=" * 60)

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"PR-AUC:  {pr_auc:.4f}"
        )

        print("\nROC curve saved to:")
        print(
            ROC_FILE
        )

        print("\nPrecision-Recall curve saved to:")
        print(
            PR_FILE
        )

        print(
            "\nFinal curve generation completed successfully."
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