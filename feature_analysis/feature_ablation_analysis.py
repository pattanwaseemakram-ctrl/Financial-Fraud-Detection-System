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
    "feature_analysis",
    "feature_ablation_results"
)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "feature_ablation_results.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "feature_ablation_summary.txt"
)


# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

TARGET_COLUMN = "Suspicious Activity Flag"


# ---------------------------------------------------------
# Features identified as redundant/useless
# ---------------------------------------------------------

REDUNDANT_FEATURES = [
    "Location_Deviation_High_Amount",
    "Amount_Percent_of_Balance"
]


# ---------------------------------------------------------
# Interaction features
# ---------------------------------------------------------

INTERACTION_FEATURES = [
    "Spending_Deviation_Amount",
    "High_Amount_Unusual_Hour",
    "Location_Deviation",
    "Unusual_Hour"
]


# ---------------------------------------------------------
# Strong signal features
# ---------------------------------------------------------

STRONG_SIGNAL_FEATURES = [
    "Transaction_Hour",
    "Spending_Deviation_Amount",
    "Account Balance",
    "Amount",
    "Transaction_Month",
    "Login Location Match",
    "High_Value_Transaction",
    "User Device Recognition",
    "Known Threat Flag",
    "Location_California",
    "Location_Florida",
    "Location_New York",
    "Location_Texas",
    "Type_Deposit",
    "Type_Transfer",
    "Type_Withdrawal"
]


# ---------------------------------------------------------
# Create model
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
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

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
        # Check target
        # -------------------------------------------------

        if TARGET_COLUMN not in df.columns:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                "was not found."
            )

        # -------------------------------------------------
        # Separate X and y
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
        # Create feature sets
        # -------------------------------------------------

        feature_sets = {}

        # -------------------------------------------------
        # 1. Current improved features
        # -------------------------------------------------

        feature_sets[
            "Current Improved Features"
        ] = list(
            X.columns
        )

        # -------------------------------------------------
        # 2. Remove redundant/useless features
        # -------------------------------------------------

        cleaned_features = [
            feature
            for feature in X.columns
            if feature not in REDUNDANT_FEATURES
        ]

        feature_sets[
            "Cleaned Features"
        ] = cleaned_features

        # -------------------------------------------------
        # 3. Remove interaction features
        # -------------------------------------------------

        no_interaction_features = [
            feature
            for feature in cleaned_features
            if feature not in INTERACTION_FEATURES
        ]

        feature_sets[
            "Without Interaction Features"
        ] = no_interaction_features

        # -------------------------------------------------
        # 4. Strong signal features
        # -------------------------------------------------

        strong_features = [
            feature
            for feature in STRONG_SIGNAL_FEATURES
            if feature in X.columns
        ]

        feature_sets[
            "Strong Signal Features"
        ] = strong_features

        # -------------------------------------------------
        # Run experiments
        # -------------------------------------------------

        results = []

        print("\n")
        print("=" * 70)
        print("FEATURE ABLATION ANALYSIS")
        print("=" * 70)

        for feature_set_name, features in feature_sets.items():

            print("\n")
            print("-" * 70)

            print(
                f"Testing: {feature_set_name}"
            )

            print(
                f"Number of features: {len(features)}"
            )

            print(
                "Features:"
            )

            print(
                features
            )

            # ---------------------------------------------
            # Select feature columns
            # ---------------------------------------------

            X_train_subset = X_train[
                features
            ].copy()

            X_test_subset = X_test[
                features
            ].copy()

            # ---------------------------------------------
            # Create model
            # ---------------------------------------------

            model = create_model()

            # ---------------------------------------------
            # Evaluate
            # ---------------------------------------------

            metrics = evaluate_model(
                model,
                X_train_subset,
                X_test_subset,
                y_train,
                y_test
            )

            # ---------------------------------------------
            # Store results
            # ---------------------------------------------

            result = {
                "Feature Set":
                    feature_set_name,

                "Number of Features":
                    len(features),

                **metrics
            }

            results.append(
                result
            )

            # ---------------------------------------------
            # Display results
            # ---------------------------------------------

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
        # Results DataFrame
        # -------------------------------------------------

        results_df = pd.DataFrame(
            results
        )

        # -------------------------------------------------
        # Sort by F1
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
        # Save CSV
        # -------------------------------------------------

        results_df.to_csv(
            RESULT_FILE,
            index=False
        )

        # -------------------------------------------------
        # Display final comparison
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FEATURE SET COMPARISON")
        print("=" * 70)

        display_columns = [
            "Feature Set",
            "Number of Features",
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
        # Best feature set
        # -------------------------------------------------

        best_result = results_df.iloc[0]

        best_feature_set = (
            best_result["Feature Set"]
        )

        best_f1 = (
            best_result["F1"]
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
                "FEATURE ABLATION ANALYSIS SUMMARY\n"
            )

            file.write(
                "=" * 70
                + "\n\n"
            )

            file.write(
                "Model:\n"
            )

            file.write(
                "StandardScaler -> "
                "RandomOverSampler -> "
                "LogisticRegression\n\n"
            )

            file.write(
                "ROS sampling strategy: 0.5\n"
            )

            file.write(
                "Logistic Regression C: 0.0005\n"
            )

            file.write(
                "Class Weight: balanced\n\n"
            )

            file.write(
                "Feature Set Comparison:\n\n"
            )

            file.write(
                comparison_df.to_string(
                    index=False
                )
            )

            file.write(
                "\n\nBest Feature Set:\n"
            )

            file.write(
                f"{best_feature_set}\n"
            )

            file.write(
                f"F1 Score: {best_f1:.4f}\n"
            )

            file.write(
                f"PR-AUC: {best_pr_auc:.4f}\n"
            )

        # -------------------------------------------------
        # Completion
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FEATURE ABLATION ANALYSIS COMPLETED")
        print("=" * 70)

        print(
            f"\nBest Feature Set: "
            f"{best_feature_set}"
        )

        print(
            f"Best F1 Score: "
            f"{best_f1 * 100:.2f}%"
        )

        print(
            f"Best PR-AUC: "
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

    except Exception as e:

        print(
            "\nUnexpected error:"
        )

        print(e)


if __name__ == "__main__":
    main()