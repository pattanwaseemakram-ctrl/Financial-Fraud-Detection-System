import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler


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
# Output directory
# ---------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "final_finetuning",
    "model_comparison_results"
)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison_results.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison_summary.txt"
)


# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

TARGET_COLUMN = "Suspicious Activity Flag"


# ---------------------------------------------------------
# Create Logistic Regression pipeline
# ---------------------------------------------------------

def create_logistic_regression():

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
# Create Random Forest pipeline
# ---------------------------------------------------------

def create_random_forest():

    model = Pipeline(
        steps=[
            (
                "oversampler",
                RandomOverSampler(
                    sampling_strategy=0.5,
                    random_state=42
                )
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    return model


# ---------------------------------------------------------
# Create XGBoost pipeline
# ---------------------------------------------------------

def create_xgboost():

    model = Pipeline(
        steps=[
            (
                "oversampler",
                RandomOverSampler(
                    sampling_strategy=0.5,
                    random_state=42
                )
            ),

            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    return model


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test
    )

    # Probability
    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
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

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    return {
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    try:

        # -------------------------------------------------
        # Create output directory
        # -------------------------------------------------

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

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
            f"\nDataset shape: {df.shape}"
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
        # Separate features and target
        # -------------------------------------------------

        X = df.drop(
            TARGET_COLUMN,
            axis=1
        )

        y = df[
            TARGET_COLUMN
        ]

        print(
            f"Feature shape: {X.shape}"
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
            f"\nTraining shape: {X_train.shape}"
        )

        print(
            f"Testing shape: {X_test.shape}"
        )

        # -------------------------------------------------
        # Model collection
        # -------------------------------------------------

        models = {
            "Logistic Regression":
                create_logistic_regression(),

            "Random Forest":
                create_random_forest(),

            "XGBoost":
                create_xgboost()
        }

        # -------------------------------------------------
        # Run model comparison
        # -------------------------------------------------

        results = []

        print("\n")
        print("=" * 75)
        print("FINAL MODEL COMPARISON")
        print("=" * 75)

        for model_name, model in models.items():

            print("\n")
            print("-" * 75)

            print(
                f"Testing model: {model_name}"
            )

            metrics = evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test
            )

            result = {
                "Model": model_name,
                **metrics
            }

            results.append(
                result
            )

            print("\nResults:")

            print(
                f"Accuracy  : "
                f"{metrics['Accuracy'] * 100:.2f}%"
            )

            print(
                f"Precision : "
                f"{metrics['Precision'] * 100:.2f}%"
            )

            print(
                f"Recall    : "
                f"{metrics['Recall'] * 100:.2f}%"
            )

            print(
                f"F1 Score  : "
                f"{metrics['F1'] * 100:.2f}%"
            )

            print(
                f"ROC-AUC   : "
                f"{metrics['ROC-AUC']:.4f}"
            )

            print(
                f"PR-AUC    : "
                f"{metrics['PR-AUC']:.4f}"
            )

            print(
                f"False Positives : "
                f"{metrics['False Positives']}"
            )

            print(
                f"False Negatives : "
                f"{metrics['False Negatives']}"
            )

            print(
                f"True Positives  : "
                f"{metrics['True Positives']}"
            )

            print(
                f"True Negatives  : "
                f"{metrics['True Negatives']}"
            )

        # -------------------------------------------------
        # Create results DataFrame
        # -------------------------------------------------

        results_df = pd.DataFrame(
            results
        )

        # -------------------------------------------------
        # Sort by F1 score
        # -------------------------------------------------

        results_df = (
            results_df
            .sort_values(
                by="F1",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        # -------------------------------------------------
        # Save results
        # -------------------------------------------------

        results_df.to_csv(
            RESULT_FILE,
            index=False
        )

        # -------------------------------------------------
        # Display comparison table
        # -------------------------------------------------

        print("\n")
        print("=" * 75)
        print("MODEL COMPARISON")
        print("=" * 75)

        display_columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC",
            "False Positives",
            "False Negatives"
        ]

        comparison_df = results_df[
            display_columns
        ].copy()

        print(
            comparison_df.to_string(
                index=False
            )
        )

        # -------------------------------------------------
        # Best model based on F1
        # -------------------------------------------------

        best_result = results_df.iloc[0]

        best_model = (
            best_result["Model"]
        )

        best_f1 = (
            best_result["F1"]
        )

        best_precision = (
            best_result["Precision"]
        )

        best_recall = (
            best_result["Recall"]
        )

        best_roc_auc = (
            best_result["ROC-AUC"]
        )

        best_pr_auc = (
            best_result["PR-AUC"]
        )

        # -------------------------------------------------
        # Save summary
        # -------------------------------------------------

        with open(
            SUMMARY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "FINAL MODEL COMPARISON SUMMARY\n"
            )

            file.write(
                "=" * 75
                + "\n\n"
            )

            file.write(
                "Dataset:\n"
            )

            file.write(
                "encoded_transactions_improved.csv\n\n"
            )

            file.write(
                "Train/Test Split:\n"
            )

            file.write(
                "80% Training / 20% Testing\n"
            )

            file.write(
                "random_state = 42\n\n"
            )

            file.write(
                "Balancing Method:\n"
            )

            file.write(
                "RandomOverSampler "
                "(sampling_strategy=0.5)\n\n"
            )

            file.write(
                "Models Compared:\n"
            )

            file.write(
                "1. Logistic Regression\n"
            )

            file.write(
                "2. Random Forest\n"
            )

            file.write(
                "3. XGBoost\n\n"
            )

            file.write(
                "Model Comparison:\n\n"
            )

            file.write(
                comparison_df.to_string(
                    index=False
                )
            )

            file.write(
                "\n\nModel with highest F1:\n"
            )

            file.write(
                f"{best_model}\n\n"
            )

            file.write(
                f"F1 Score   : "
                f"{best_f1 * 100:.2f}%\n"
            )

            file.write(
                f"Precision  : "
                f"{best_precision * 100:.2f}%\n"
            )

            file.write(
                f"Recall     : "
                f"{best_recall * 100:.2f}%\n"
            )

            file.write(
                f"ROC-AUC    : "
                f"{best_roc_auc:.4f}\n"
            )

            file.write(
                f"PR-AUC     : "
                f"{best_pr_auc:.4f}\n"
            )

        # -------------------------------------------------
        # Final output
        # -------------------------------------------------

        print("\n")
        print("=" * 75)
        print("FINAL MODEL COMPARISON COMPLETED")
        print("=" * 75)

        print(
            f"\nHighest F1 model: "
            f"{best_model}"
        )

        print(
            f"F1 Score: "
            f"{best_f1 * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{best_precision * 100:.2f}%"
        )

        print(
            f"Recall: "
            f"{best_recall * 100:.2f}%"
        )

        print(
            f"ROC-AUC: "
            f"{best_roc_auc:.4f}"
        )

        print(
            f"PR-AUC: "
            f"{best_pr_auc:.4f}"
        )

        print("\nResults saved to:")

        print(
            RESULT_FILE
        )

        print(
            SUMMARY_FILE
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

    except ImportError as e:

        print(
            "\nImport error:"
        )

        print(e)

    except Exception as e:

        print(
            "\nUnexpected error:"
        )

        print(e)


if __name__ == "__main__":
    main()