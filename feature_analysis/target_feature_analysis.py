import os
import pandas as pd
import numpy as np

from sklearn.feature_selection import (
    mutual_info_classif
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    roc_auc_score
)


# ---------------------------------------------------------
# Input dataset
# ---------------------------------------------------------

INPUT_FILE = (
    "Dataset/processed data/encoded_transactions_improved.csv"
)

# Output directory
OUTPUT_DIR = "feature_analysis/target_relationship_results"


# Output files
FEATURE_RELATIONSHIP_FILE = (
    "feature_analysis/target_relationship_results/"
    "target_feature_relationship.csv"
)

CATEGORY_ANALYSIS_FILE = (
    "feature_analysis/target_relationship_results/"
    "category_fraud_rates.csv"
)

SUMMARY_FILE = (
    "feature_analysis/target_relationship_results/"
    "target_feature_analysis_summary.txt"
)


# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

TARGET_COLUMN = "Suspicious Activity Flag"


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

        print("Loading improved encoded dataset...")

        df = pd.read_csv(
            INPUT_FILE
        )

        print(
            "Dataset loaded successfully."
        )

        print("\nDataset shape:")
        print(
            df.shape
        )

        # ---------------------------------------------------------
        # Check target column
        # ---------------------------------------------------------

        if TARGET_COLUMN not in df.columns:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                "was not found."
            )

        # ---------------------------------------------------------
        # Separate features and target
        # ---------------------------------------------------------

        X = df.drop(
            TARGET_COLUMN,
            axis=1
        )

        y = df[
            TARGET_COLUMN
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
        # Create development split
        #
        # Analysis metrics are calculated on training data only
        # where possible.
        # ---------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print("\nTraining data shape:")
        print(
            X_train.shape
        )

        print("\nTesting data shape:")
        print(
            X_test.shape
        )

        # ---------------------------------------------------------
        # Random Forest feature importance
        # ---------------------------------------------------------

        print(
            "\nCalculating Random Forest feature importance..."
        )

        random_forest = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

        random_forest.fit(
            X_train,
            y_train
        )

        rf_importance = pd.Series(
            random_forest.feature_importances_,
            index=X_train.columns
        )

        # ---------------------------------------------------------
        # Mutual information
        # ---------------------------------------------------------

        print(
            "\nCalculating mutual information..."
        )

        mi_values = mutual_info_classif(
            X_train,
            y_train,
            random_state=42
        )

        mutual_information = pd.Series(
            mi_values,
            index=X_train.columns
        )

        # ---------------------------------------------------------
        # Calculate class-wise statistics
        # ---------------------------------------------------------

        print(
            "\nCalculating class-wise statistics..."
        )

        normal_data = X_train[
            y_train == 0
        ]

        suspicious_data = X_train[
            y_train == 1
        ]

        relationship_results = []

        for feature in X_train.columns:

            normal_mean = (
                normal_data[feature]
                .mean()
            )

            suspicious_mean = (
                suspicious_data[feature]
                .mean()
            )

            normal_median = (
                normal_data[feature]
                .median()
            )

            suspicious_median = (
                suspicious_data[feature]
                .median()
            )

            normal_std = (
                normal_data[feature]
                .std()
            )

            suspicious_std = (
                suspicious_data[feature]
                .std()
            )

            mean_difference = (
                suspicious_mean
                - normal_mean
            )

            absolute_mean_difference = abs(
                mean_difference
            )

            # -----------------------------------------------------
            # Standardized mean difference
            #
            # This helps compare differences across features that
            # have different scales.
            # -----------------------------------------------------

            pooled_std = np.sqrt(
                (
                    normal_std ** 2
                    +
                    suspicious_std ** 2
                ) / 2
            )

            if pooled_std != 0:
                standardized_difference = (
                    mean_difference
                    / pooled_std
                )
            else:
                standardized_difference = 0

            relationship_results.append({
                "Feature": feature,
                "RF Importance": rf_importance[feature],
                "Mutual Information":
                    mutual_information[feature],
                "Normal Mean": normal_mean,
                "Suspicious Mean":
                    suspicious_mean,
                "Normal Median":
                    normal_median,
                "Suspicious Median":
                    suspicious_median,
                "Normal Std":
                    normal_std,
                "Suspicious Std":
                    suspicious_std,
                "Mean Difference":
                    mean_difference,
                "Absolute Mean Difference":
                    absolute_mean_difference,
                "Standardized Difference":
                    standardized_difference
            })

        relationship_df = pd.DataFrame(
            relationship_results
        )

        # ---------------------------------------------------------
        # Create rankings
        # ---------------------------------------------------------

        relationship_df["RF Rank"] = (
            relationship_df[
                "RF Importance"
            ]
            .rank(
                ascending=False,
                method="min"
            )
        )

        relationship_df["MI Rank"] = (
            relationship_df[
                "Mutual Information"
            ]
            .rank(
                ascending=False,
                method="min"
            )
        )

        relationship_df["Difference Rank"] = (
            relationship_df[
                "Absolute Mean Difference"
            ]
            .rank(
                ascending=False,
                method="min"
            )
        )

        relationship_df["Standardized Rank"] = (
            relationship_df[
                "Standardized Difference"
            ]
            .abs()
            .rank(
                ascending=False,
                method="min"
            )
        )

        # ---------------------------------------------------------
        # Calculate overall signal score
        # ---------------------------------------------------------

        relationship_df["Average Rank"] = (
            relationship_df["RF Rank"]
            + relationship_df["MI Rank"]
            + relationship_df["Difference Rank"]
            + relationship_df["Standardized Rank"]
        ) / 4

        relationship_df = (
            relationship_df
            .sort_values(
                by="Average Rank",
                ascending=True
            )
            .reset_index(
                drop=True
            )
        )

        # ---------------------------------------------------------
        # Create categorical fraud-rate analysis
        #
        # This uses the original categorical columns if they
        # are present in the improved dataset.
        # ---------------------------------------------------------

        category_results = []

        original_categorical_columns = [
            "Type",
            "Location",
            "Transaction_Weekday"
        ]

        # The improved encoded dataset no longer contains the
        # original categorical columns, so analyse one-hot
        # categorical features instead.
        categorical_feature_names = [
            column
            for column in X.columns
            if (
                column.startswith("Type_")
                or column.startswith("Location_")
                or column.startswith(
                    "Transaction_Weekday_"
                )
            )
        ]

        for feature in categorical_feature_names:

            total_count = (
                df[feature] == 1
            ).sum()

            suspicious_count = (
                (
                    df[feature] == 1
                )
                &
                (
                    df[TARGET_COLUMN] == 1
                )
            ).sum()

            normal_count = (
                (
                    df[feature] == 1
                )
                &
                (
                    df[TARGET_COLUMN] == 0
                )
            ).sum()

            if total_count > 0:

                fraud_rate = (
                    suspicious_count
                    / total_count
                )

            else:

                fraud_rate = 0

            category_results.append({
                "Feature": feature,
                "Total Occurrences":
                    total_count,
                "Normal Count":
                    normal_count,
                "Suspicious Count":
                    suspicious_count,
                "Suspicious Rate":
                    fraud_rate
            })

        category_df = pd.DataFrame(
            category_results
        )

        if not category_df.empty:

            category_df = (
                category_df
                .sort_values(
                    by="Suspicious Rate",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

        # ---------------------------------------------------------
        # Save relationship results
        # ---------------------------------------------------------

        relationship_df.to_csv(
            FEATURE_RELATIONSHIP_FILE,
            index=False
        )

        category_df.to_csv(
            CATEGORY_ANALYSIS_FILE,
            index=False
        )

        # ---------------------------------------------------------
        # Display strongest features
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("TOP FEATURES BY OVERALL SIGNAL")
        print("=" * 60)

        print(
            relationship_df[
                [
                    "Feature",
                    "RF Importance",
                    "Mutual Information",
                    "Absolute Mean Difference",
                    "Standardized Difference",
                    "Average Rank"
                ]
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Display strongest mutual-information features
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("TOP FEATURES BY MUTUAL INFORMATION")
        print("=" * 60)

        print(
            relationship_df[
                [
                    "Feature",
                    "Mutual Information"
                ]
            ]
            .sort_values(
                by="Mutual Information",
                ascending=False
            )
            .head(15)
            .to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Display strongest standardized differences
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("TOP FEATURES BY STANDARDIZED DIFFERENCE")
        print("=" * 60)

        print(
            relationship_df[
                [
                    "Feature",
                    "Normal Mean",
                    "Suspicious Mean",
                    "Standardized Difference"
                ]
            ]
            .sort_values(
                by="Standardized Difference",
                key=lambda column:
                    column.abs(),
                ascending=False
            )
            .head(15)
            .to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Display categorical suspicious rates
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("CATEGORICAL FEATURE SUSPICIOUS RATES")
        print("=" * 60)

        if not category_df.empty:

            print(
                category_df.head(20).to_string(
                    index=False
                )
            )

        else:

            print(
                "No categorical one-hot features found."
            )

        # ---------------------------------------------------------
        # Save summary report
        # ---------------------------------------------------------

        with open(
            SUMMARY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "TARGET-FEATURE RELATIONSHIP ANALYSIS\n"
            )

            file.write(
                "=" * 60
                + "\n\n"
            )

            file.write(
                f"Dataset Shape: {df.shape}\n"
            )

            file.write(
                f"Number of Features: {X.shape[1]}\n"
            )

            file.write(
                f"Training Samples: "
                f"{len(X_train)}\n"
            )

            file.write(
                f"Testing Samples: "
                f"{len(X_test)}\n\n"
            )

            file.write(
                "Top Features by Overall Signal:\n\n"
            )

            file.write(
                relationship_df[
                    [
                        "Feature",
                        "RF Importance",
                        "Mutual Information",
                        "Absolute Mean Difference",
                        "Standardized Difference",
                        "Average Rank"
                    ]
                ]
                .head(15)
                .to_string(
                    index=False
                )
            )

            file.write(
                "\n\nTop Features by Mutual Information:\n\n"
            )

            file.write(
                relationship_df[
                    [
                        "Feature",
                        "Mutual Information"
                    ]
                ]
                .sort_values(
                    by="Mutual Information",
                    ascending=False
                )
                .head(15)
                .to_string(
                    index=False
                )
            )

            file.write(
                "\n\nTop Features by Standardized Difference:\n\n"
            )

            file.write(
                relationship_df[
                    [
                        "Feature",
                        "Normal Mean",
                        "Suspicious Mean",
                        "Standardized Difference"
                    ]
                ]
                .sort_values(
                    by="Standardized Difference",
                    key=lambda column:
                        column.abs(),
                    ascending=False
                )
                .head(15)
                .to_string(
                    index=False
                )
            )

            if not category_df.empty:

                file.write(
                    "\n\nCategorical Feature "
                    "Suspicious Rates:\n\n"
                )

                file.write(
                    category_df.head(20)
                    .to_string(
                        index=False
                    )
                )

        # ---------------------------------------------------------
        # Display output files
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("TARGET-FEATURE ANALYSIS COMPLETED")
        print("=" * 60)

        print("\nFiles saved:")

        print(
            FEATURE_RELATIONSHIP_FILE
        )

        print(
            CATEGORY_ANALYSIS_FILE
        )

        print(
            SUMMARY_FILE
        )

        print(
            "\nTarget-feature analysis "
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