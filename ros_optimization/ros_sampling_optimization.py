import os
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# ---------------------------------------------------------
# Input dataset
# ---------------------------------------------------------

INPUT_FILE = (
    "Dataset/processed data/encoded_transactions.csv"
)

# Output directory
OUTPUT_DIR = "ros_optimization/results"

# Output results file
OUTPUT_FILE = (
    "ros_optimization/results/"
    "ros_sampling_optimization_results.csv"
)


# ---------------------------------------------------------
# ROS sampling ratios to test
# ---------------------------------------------------------

ROS_RATIOS = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.7,
    1.0
]


# ---------------------------------------------------------
# Class weight settings to test
# ---------------------------------------------------------

CLASS_WEIGHTS = [
    None,
    "balanced"
]


# ---------------------------------------------------------
# Logistic Regression parameters
#
# We use the best C and solver found in our previous
# end-to-end ROS + Logistic Regression experiment.
# ---------------------------------------------------------

LOGISTIC_REGRESSION_PARAMS = {
    "C": 0.001,
    "solver": "liblinear",
    "max_iter": 1000
}


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

        print("\nFeature shape:")
        print(X.shape)

        print("\nTarget distribution:")
        print(y.value_counts())

        # ---------------------------------------------------------
        # Create train/test split
        #
        # The test set will NOT be used for selecting the best
        # ROS configuration.
        # ---------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print("\nTraining data shape:")
        print(X_train.shape)

        print("\nTesting data shape:")
        print(X_test.shape)

        # ---------------------------------------------------------
        # Create stratified 5-fold cross-validation
        # ---------------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        # ---------------------------------------------------------
        # Define scoring metrics
        # ---------------------------------------------------------

        scoring = {
            "precision": make_scorer(
                precision_score,
                zero_division=0
            ),

            "recall": make_scorer(
                recall_score,
                zero_division=0
            ),

            "f1": make_scorer(
                f1_score,
                zero_division=0
            ),

            "roc_auc": "roc_auc",

            "pr_auc": make_scorer(
                average_precision_score,
                response_method="predict_proba"
            )
        }

        # ---------------------------------------------------------
        # Store experiment results
        # ---------------------------------------------------------

        results = []

        total_experiments = (
            len(ROS_RATIOS)
            * len(CLASS_WEIGHTS)
        )

        experiment_number = 0

        print("\n")
        print("=" * 60)
        print("ROS SAMPLING-RATIO OPTIMIZATION")
        print("=" * 60)

        print(
            f"\nTotal configurations: "
            f"{total_experiments}"
        )

        # ---------------------------------------------------------
        # Run all ROS configurations
        # ---------------------------------------------------------

        for ratio in ROS_RATIOS:

            for class_weight in CLASS_WEIGHTS:

                experiment_number += 1

                print("\n")
                print("-" * 60)

                print(
                    f"Experiment "
                    f"{experiment_number}/"
                    f"{total_experiments}"
                )

                print(
                    f"ROS Sampling Ratio: {ratio}"
                )

                print(
                    f"Class Weight: {class_weight}"
                )

                # -------------------------------------------------
                # Create ROS sampler
                # -------------------------------------------------

                ros = RandomOverSampler(
                    sampling_strategy=ratio,
                    random_state=42
                )

                # -------------------------------------------------
                # Create Logistic Regression model
                # -------------------------------------------------

                model = LogisticRegression(
                    C=LOGISTIC_REGRESSION_PARAMS["C"],
                    solver=LOGISTIC_REGRESSION_PARAMS["solver"],
                    max_iter=LOGISTIC_REGRESSION_PARAMS["max_iter"],
                    class_weight=class_weight
                )

                # -------------------------------------------------
                # Create pipeline
                #
                # ROS is applied separately inside every
                # cross-validation training fold.
                # -------------------------------------------------

                pipeline = Pipeline([
                    (
                        "ros",
                        ros
                    ),
                    (
                        "model",
                        model
                    )
                ])

                # -------------------------------------------------
                # Perform 5-fold cross-validation
                # -------------------------------------------------

                cv_results = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=False
                )

                # -------------------------------------------------
                # Calculate mean CV metrics
                # -------------------------------------------------

                mean_precision = (
                    cv_results[
                        "test_precision"
                    ].mean()
                )

                mean_recall = (
                    cv_results[
                        "test_recall"
                    ].mean()
                )

                mean_f1 = (
                    cv_results[
                        "test_f1"
                    ].mean()
                )

                mean_roc_auc = (
                    cv_results[
                        "test_roc_auc"
                    ].mean()
                )

                mean_pr_auc = (
                    cv_results[
                        "test_pr_auc"
                    ].mean()
                )

                # -------------------------------------------------
                # Display results
                # -------------------------------------------------

                print(
                    f"CV Precision: {mean_precision:.4f}"
                )

                print(
                    f"CV Recall:    {mean_recall:.4f}"
                )

                print(
                    f"CV F1:        {mean_f1:.4f}"
                )

                print(
                    f"CV ROC-AUC:   {mean_roc_auc:.4f}"
                )

                print(
                    f"CV PR-AUC:    {mean_pr_auc:.4f}"
                )

                # -------------------------------------------------
                # Save result
                # -------------------------------------------------

                results.append({
                    "ROS Sampling Ratio": ratio,
                    "Class Weight": class_weight,
                    "CV Precision": mean_precision,
                    "CV Recall": mean_recall,
                    "CV F1": mean_f1,
                    "CV ROC-AUC": mean_roc_auc,
                    "CV PR-AUC": mean_pr_auc
                })

        # ---------------------------------------------------------
        # Create result dataframe
        # ---------------------------------------------------------

        results_df = pd.DataFrame(
            results
        )

        # ---------------------------------------------------------
        # Sort by CV F1
        #
        # F1 is our primary metric for this experiment because
        # it balances precision and recall.
        # ---------------------------------------------------------

        results_df = results_df.sort_values(
            by="CV F1",
            ascending=False
        ).reset_index(
            drop=True
        )

        # ---------------------------------------------------------
        # Save results
        # ---------------------------------------------------------

        results_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        # ---------------------------------------------------------
        # Display complete comparison
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("ROS OPTIMIZATION RESULTS")
        print("=" * 60)

        print(
            results_df.to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Display best configuration
        # ---------------------------------------------------------

        best_result = results_df.iloc[0]

        print("\n")
        print("=" * 60)
        print("BEST ROS CONFIGURATION")
        print("=" * 60)

        print(
            f"ROS Sampling Ratio: "
            f"{best_result['ROS Sampling Ratio']}"
        )

        print(
            f"Class Weight: "
            f"{best_result['Class Weight']}"
        )

        print(
            f"CV Precision: "
            f"{best_result['CV Precision']:.4f}"
        )

        print(
            f"CV Recall: "
            f"{best_result['CV Recall']:.4f}"
        )

        print(
            f"CV F1: "
            f"{best_result['CV F1']:.4f}"
        )

        print(
            f"CV ROC-AUC: "
            f"{best_result['CV ROC-AUC']:.4f}"
        )

        print(
            f"CV PR-AUC: "
            f"{best_result['CV PR-AUC']:.4f}"
        )

        print("\nResults saved to:")
        print(OUTPUT_FILE)

        print(
            "\nROS sampling-ratio optimization "
            "completed successfully."
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