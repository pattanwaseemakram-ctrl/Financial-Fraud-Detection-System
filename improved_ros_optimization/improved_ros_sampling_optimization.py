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
    average_precision_score
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# ---------------------------------------------------------
# Input dataset
# ---------------------------------------------------------

INPUT_FILE = (
    "Dataset/processed data/encoded_transactions_improved.csv"
)

# Output directory
OUTPUT_DIR = "improved_ros_optimization/results"

# Output results file
OUTPUT_FILE = (
    "improved_ros_optimization/results/"
    "improved_ros_sampling_optimization_results.csv"
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
# We use a wider C range because feature engineering
# improved the model and we want to test different
# regularization strengths.
# ---------------------------------------------------------

C_VALUES = [
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1,
    5,
    10
]


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
        # The test set is not used for selecting the best
        # configuration.
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
        # Create 5-fold stratified cross-validation
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

        # 6 ROS ratios × 2 class weights × 11 C values
        total_experiments = (
            len(ROS_RATIOS)
            * len(CLASS_WEIGHTS)
            * len(C_VALUES)
        )

        experiment_number = 0

        print("\n")
        print("=" * 60)
        print("IMPROVED ROS + LOGISTIC REGRESSION OPTIMIZATION")
        print("=" * 60)

        print(
            f"\nTotal configurations: "
            f"{total_experiments}"
        )

        # ---------------------------------------------------------
        # Run all configurations
        # ---------------------------------------------------------

        for ratio in ROS_RATIOS:

            for class_weight in CLASS_WEIGHTS:

                for c_value in C_VALUES:

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

                    print(
                        f"C: {c_value}"
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
                        C=c_value,
                        solver="liblinear",
                        class_weight=class_weight,
                        max_iter=1000
                    )

                    # -------------------------------------------------
                    # Create pipeline
                    #
                    # ROS is applied inside each training fold.
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
                    # Store result
                    # -------------------------------------------------

                    results.append({
                        "ROS Sampling Ratio": ratio,
                        "Class Weight": class_weight,
                        "C": c_value,
                        "CV Precision": mean_precision,
                        "CV Recall": mean_recall,
                        "CV F1": mean_f1,
                        "CV ROC-AUC": mean_roc_auc,
                        "CV PR-AUC": mean_pr_auc
                    })

        # ---------------------------------------------------------
        # Create results dataframe
        # ---------------------------------------------------------

        results_df = pd.DataFrame(
            results
        )

        # ---------------------------------------------------------
        # Sort by CV F1
        #
        # F1 is the primary metric for this experiment.
        # ---------------------------------------------------------

        results_df = results_df.sort_values(
            by="CV F1",
            ascending=False
        ).reset_index(
            drop=True
        )

        # ---------------------------------------------------------
        # Save complete experiment results
        # ---------------------------------------------------------

        results_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        # ---------------------------------------------------------
        # Display top 10 configurations
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("TOP 10 ROS CONFIGURATIONS")
        print("=" * 60)

        print(
            results_df.head(10).to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Display best configuration
        # ---------------------------------------------------------

        best_result = results_df.iloc[0]

        print("\n")
        print("=" * 60)
        print("BEST IMPROVED ROS CONFIGURATION")
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
            f"C: "
            f"{best_result['C']}"
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
            "\nImproved ROS optimization "
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