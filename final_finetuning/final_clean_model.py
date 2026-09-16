import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import joblib


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
# Output directories
# ---------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "final_finetuning",
    "final_clean_model_results"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    OUTPUT_DIR,
    "results"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "final_clean_logistic_model.pkl"
)

METRICS_FILE = os.path.join(
    RESULT_DIR,
    "final_clean_model_metrics.csv"
)

REPORT_FILE = os.path.join(
    RESULT_DIR,
    "final_clean_model_report.txt"
)

FEATURE_FILE = os.path.join(
    RESULT_DIR,
    "final_clean_features.txt"
)

CONFUSION_MATRIX_FILE = os.path.join(
    RESULT_DIR,
    "confusion_matrix.csv"
)


# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

TARGET_COLUMN = "Suspicious Activity Flag"


# ---------------------------------------------------------
# Features to remove
# ---------------------------------------------------------

CONSTANT_FEATURES = [
    "Transaction_Month",
    "High_Value_Transaction",
    "Low_Balance",
    "High_Amount_Unusual_Hour",
    "Location_Deviation_High_Amount"
]


REDUNDANT_FEATURES = [
    "Amount_Percent_of_Balance"
]


# ---------------------------------------------------------
# Create final model pipeline
# ---------------------------------------------------------

def create_model():

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),

            (
                "oversampler",
                RandomOverSampler(
                    sampling_strategy=0.5,
                    random_state=42
                )
            ),

            (
                "classifier",
                LogisticRegression(
                    C=0.0005,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42
                )
            )
        ]
    )

    return model


# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------

def main():

    try:

        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------

        print(
            "Loading improved encoded dataset..."
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
        # Validate target
        # -------------------------------------------------

        if TARGET_COLUMN not in df.columns:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                "was not found."
            )

        # -------------------------------------------------
        # Remove constant features
        # -------------------------------------------------

        features_to_remove = (
            CONSTANT_FEATURES
            + REDUNDANT_FEATURES
        )

        existing_features_to_remove = [
            feature
            for feature in features_to_remove
            if feature in df.columns
        ]

        df_clean = df.drop(
            columns=existing_features_to_remove
        )

        print(
            "\nFeatures removed:"
        )

        for feature in existing_features_to_remove:

            print(
                f"- {feature}"
            )

        print(
            f"\nCleaned dataset shape: "
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

        print(
            "\nTarget distribution:"
        )

        print(
            y.value_counts()
        )

        # -------------------------------------------------
        # Train-test split
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print(
            f"\nTraining data: "
            f"{X_train.shape}"
        )

        print(
            f"Testing data: "
            f"{X_test.shape}"
        )

        # -------------------------------------------------
        # Create final model
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("TRAINING FINAL CLEAN MODEL")
        print("=" * 70)

        model = create_model()

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        print(
            "\nModel training completed."
        )

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------

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
        print("FINAL CLEAN MODEL RESULTS")
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

        print(
            f"\nTrue Negatives  : {tn}"
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

        print("\n")
        print("=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        # -------------------------------------------------
        # Save model
        # -------------------------------------------------

        joblib.dump(
            model,
            MODEL_FILE
        )

        print(
            f"Model saved to:\n"
            f"{MODEL_FILE}"
        )

        # -------------------------------------------------
        # Save metrics
        # -------------------------------------------------

        metrics_df = pd.DataFrame([
            {
                "Model": "Final Clean Logistic Regression",
                "Features": X.shape[1],
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
                "True Negatives": tn,
                "False Positives": fp,
                "False Negatives": fn,
                "True Positives": tp
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
        # Save final feature list
        # -------------------------------------------------

        with open(
            FEATURE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "FINAL CLEAN MODEL FEATURES\n"
            )

            file.write(
                "=" * 50
                + "\n\n"
            )

            file.write(
                f"Total features: "
                f"{len(X.columns)}\n\n"
            )

            file.write(
                "Features used:\n"
            )

            for feature in X.columns:

                file.write(
                    f"- {feature}\n"
                )

            file.write(
                "\nFeatures removed:\n"
            )

            for feature in existing_features_to_remove:

                file.write(
                    f"- {feature}\n"
                )

        # -------------------------------------------------
        # Save detailed report
        # -------------------------------------------------

        with open(
            REPORT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "FINAL CLEAN MODEL EVALUATION REPORT\n"
            )

            file.write(
                "=" * 70
                + "\n\n"
            )

            file.write(
                "Model: Logistic Regression\n"
            )

            file.write(
                "Scaling: StandardScaler\n"
            )

            file.write(
                "Balancing: RandomOverSampler\n"
            )

            file.write(
                "ROS sampling strategy: 0.5\n"
            )

            file.write(
                "Class weight: balanced\n"
            )

            file.write(
                "Logistic Regression C: 0.0005\n"
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
                f"Original dataset shape: "
                f"{df.shape}\n"
            )

            file.write(
                f"Cleaned dataset shape: "
                f"{df_clean.shape}\n"
            )

            file.write(
                f"Final feature count: "
                f"{X.shape[1]}\n\n"
            )

            file.write(
                "Removed Features:\n"
            )

            for feature in existing_features_to_remove:

                file.write(
                    f"- {feature}\n"
                )

            file.write(
                "\nFinal Evaluation Metrics:\n\n"
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
                classification_report(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )

        # -------------------------------------------------
        # Final message
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL CLEAN MODEL TRAINING COMPLETED")
        print("=" * 70)

        print("\nSaved files:")

        print(
            MODEL_FILE
        )

        print(
            METRICS_FILE
        )

        print(
            REPORT_FILE
        )

        print(
            FEATURE_FILE
        )

        print(
            CONFUSION_MATRIX_FILE
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