import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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
    "data_quality_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

DUPLICATE_FILE = os.path.join(
    OUTPUT_DIR,
    "duplicate_analysis.csv"
)

FEATURE_LABEL_FILE = os.path.join(
    OUTPUT_DIR,
    "feature_label_consistency.csv"
)

TARGET_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "target_validation_summary.txt"
)


# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

TARGET_COLUMN = "Suspicious Activity Flag"


# ---------------------------------------------------------
# Key features to inspect
# ---------------------------------------------------------

KEY_FEATURES = [
    "Amount",
    "Account Balance",
    "User Device Recognition",
    "Known Threat Flag",
    "Login Location Match",
    "Spending Pattern Deviation",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "High_Value_Transaction",
    "Low_Balance",
    "Amount_to_Balance_Ratio",
    "Spending_Deviation_Amount",
    "Location_Deviation",
    "Unusual_Hour",
    "High_Amount_Unusual_Hour",
    "Location_Deviation_High_Amount",
    "Type_Deposit",
    "Type_Transfer",
    "Type_Withdrawal",
    "Location_California",
    "Location_Florida",
    "Location_New York",
    "Location_Texas"
]


# ---------------------------------------------------------
# Main
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
        # Basic dataset information
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("DATA QUALITY & TARGET VALIDATION")
        print("=" * 70)

        print("\nColumn names:")

        for column in df.columns:

            print(
                f"- {column}"
            )

        # -------------------------------------------------
        # Target distribution
        # -------------------------------------------------

        target_counts = (
            df[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        print("\n")
        print("=" * 70)
        print("TARGET DISTRIBUTION")
        print("=" * 70)

        print(
            target_counts
        )

        normal_count = (
            target_counts.get(
                0,
                0
            )
        )

        suspicious_count = (
            target_counts.get(
                1,
                0
            )
        )

        total_count = len(df)

        normal_percentage = (
            normal_count
            / total_count
            * 100
        )

        suspicious_percentage = (
            suspicious_count
            / total_count
            * 100
        )

        print(
            f"\nNormal transactions: "
            f"{normal_count} "
            f"({normal_percentage:.2f}%)"
        )

        print(
            f"Suspicious transactions: "
            f"{suspicious_count} "
            f"({suspicious_percentage:.2f}%)"
        )

        # -------------------------------------------------
        # Missing values
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("MISSING VALUE ANALYSIS")
        print("=" * 70)

        missing_values = (
            df.isnull()
            .sum()
        )

        missing_values = (
            missing_values[
                missing_values > 0
            ]
            .sort_values(
                ascending=False
            )
        )

        if missing_values.empty:

            print(
                "No missing values found."
            )

        else:

            print(
                missing_values
            )

        # -------------------------------------------------
        # Duplicate row analysis
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("DUPLICATE ROW ANALYSIS")
        print("=" * 70)

        duplicate_rows = (
            df.duplicated()
            .sum()
        )

        print(
            f"Duplicate rows: "
            f"{duplicate_rows}"
        )

        duplicate_percentage = (
            duplicate_rows
            / total_count
            * 100
        )

        print(
            f"Duplicate percentage: "
            f"{duplicate_percentage:.2f}%"
        )

        # -------------------------------------------------
        # Duplicate analysis with target
        # -------------------------------------------------

        duplicate_groups = (
            df.groupby(
                list(
                    df.columns.drop(
                        TARGET_COLUMN
                    )
                ),
                dropna=False
            )[TARGET_COLUMN]
            .agg(
                [
                    "count",
                    "nunique"
                ]
            )
            .reset_index()
        )

        duplicate_groups = (
            duplicate_groups[
                duplicate_groups["count"] > 1
            ]
            .copy()
        )

        contradictory_groups = (
            duplicate_groups[
                duplicate_groups["nunique"] > 1
            ]
            .copy()
        )

        print("\n")
        print(
            f"Repeated feature combinations: "
            f"{len(duplicate_groups)}"
        )

        print(
            f"Repeated feature combinations "
            f"with different labels: "
            f"{len(contradictory_groups)}"
        )

        # -------------------------------------------------
        # Save duplicate analysis
        # -------------------------------------------------

        duplicate_output = pd.DataFrame({
            "Metric": [
                "Total Rows",
                "Exact Duplicate Rows",
                "Duplicate Percentage",
                "Repeated Feature Combinations",
                "Contradictory Feature Combinations"
            ],
            "Value": [
                total_count,
                duplicate_rows,
                duplicate_percentage,
                len(duplicate_groups),
                len(contradictory_groups)
            ]
        })

        duplicate_output.to_csv(
            DUPLICATE_FILE,
            index=False
        )

        # -------------------------------------------------
        # Constant feature analysis
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("CONSTANT FEATURE ANALYSIS")
        print("=" * 70)

        constant_features = []

        for column in df.columns:

            unique_count = (
                df[column]
                .nunique(
                    dropna=False
                )
            )

            if (
                column != TARGET_COLUMN
                and unique_count <= 1
            ):

                constant_features.append(
                    column
                )

        if constant_features:

            print(
                "Constant features found:"
            )

            for feature in constant_features:

                print(
                    f"- {feature}"
                )

        else:

            print(
                "No constant features found."
            )

        # -------------------------------------------------
        # Target distribution by key features
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FEATURE-TARGET CONSISTENCY ANALYSIS")
        print("=" * 70)

        feature_results = []

        available_features = [
            feature
            for feature in KEY_FEATURES
            if feature in df.columns
        ]

        for feature in available_features:

            feature_data = df[
                feature
            ]

            # ---------------------------------------------
            # Numeric / binary feature statistics
            # ---------------------------------------------

            normal_data = df.loc[
                df[TARGET_COLUMN] == 0,
                feature
            ]

            suspicious_data = df.loc[
                df[TARGET_COLUMN] == 1,
                feature
            ]

            normal_mean = (
                normal_data.mean()
            )

            suspicious_mean = (
                suspicious_data.mean()
            )

            normal_median = (
                normal_data.median()
            )

            suspicious_median = (
                suspicious_data.median()
            )

            normal_std = (
                normal_data.std()
            )

            suspicious_std = (
                suspicious_data.std()
            )

            # ---------------------------------------------
            # Mean difference
            # ---------------------------------------------

            mean_difference = (
                suspicious_mean
                - normal_mean
            )

            absolute_difference = abs(
                mean_difference
            )

            # ---------------------------------------------
            # Pooled standard deviation
            # ---------------------------------------------

            pooled_std = np.sqrt(
                (
                    normal_std ** 2
                    +
                    suspicious_std ** 2
                ) / 2
            )

            if pooled_std > 0:

                standardized_difference = (
                    mean_difference
                    / pooled_std
                )

            else:

                standardized_difference = 0

            # ---------------------------------------------
            # If feature is binary, calculate positive
            # rates in both classes.
            # ---------------------------------------------

            is_binary = (
                feature_data.nunique() <= 2
            )

            normal_positive_rate = np.nan
            suspicious_positive_rate = np.nan
            positive_rate_difference = np.nan

            if is_binary:

                normal_positive_rate = (
                    (normal_data == 1)
                    .mean()
                )

                suspicious_positive_rate = (
                    (suspicious_data == 1)
                    .mean()
                )

                positive_rate_difference = (
                    suspicious_positive_rate
                    - normal_positive_rate
                )

            # ---------------------------------------------
            # Univariate ROC-AUC
            #
            # Calculated using the complete feature column
            # only as a diagnostic of individual separation.
            # ---------------------------------------------

            try:

                if feature_data.nunique() > 1:

                    feature_auc = (
                        roc_auc_score(
                            df[TARGET_COLUMN],
                            feature_data
                        )
                    )

                    directional_auc = max(
                        feature_auc,
                        1 - feature_auc
                    )

                else:

                    feature_auc = np.nan
                    directional_auc = np.nan

            except Exception:

                feature_auc = np.nan
                directional_auc = np.nan

            feature_results.append({

                "Feature":
                    feature,

                "Unique Values":
                    feature_data.nunique(),

                "Normal Mean":
                    normal_mean,

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
                    absolute_difference,

                "Standardized Difference":
                    standardized_difference,

                "Normal Positive Rate":
                    normal_positive_rate,

                "Suspicious Positive Rate":
                    suspicious_positive_rate,

                "Positive Rate Difference":
                    positive_rate_difference,

                "Raw ROC-AUC":
                    feature_auc,

                "Directional ROC-AUC":
                    directional_auc
            })

        feature_df = pd.DataFrame(
            feature_results
        )

        # -------------------------------------------------
        # Sort by individual feature separation
        # -------------------------------------------------

        feature_df = (
            feature_df
            .sort_values(
                by="Directional ROC-AUC",
                ascending=False,
                na_position="last"
            )
            .reset_index(
                drop=True
            )
        )

        # -------------------------------------------------
        # Save feature-target analysis
        # -------------------------------------------------

        feature_df.to_csv(
            FEATURE_LABEL_FILE,
            index=False
        )

        # -------------------------------------------------
        # Display feature results
        # -------------------------------------------------

        print("\n")
        print(
            "TOP FEATURES BY DIRECTIONAL ROC-AUC"
        )

        print(
            feature_df[
                [
                    "Feature",
                    "Directional ROC-AUC",
                    "Normal Mean",
                    "Suspicious Mean",
                    "Standardized Difference"
                ]
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

        # -------------------------------------------------
        # Strongest class difference
        # -------------------------------------------------

        print("\n")
        print(
            "TOP FEATURES BY STANDARDIZED DIFFERENCE"
        )

        strongest_difference = (
            feature_df
            .assign(
                Absolute_Std_Difference=
                feature_df[
                    "Standardized Difference"
                ].abs()
            )
            .sort_values(
                by="Absolute_Std_Difference",
                ascending=False
            )
        )

        print(
            strongest_difference[
                [
                    "Feature",
                    "Normal Mean",
                    "Suspicious Mean",
                    "Standardized Difference"
                ]
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

        # -------------------------------------------------
        # Binary feature analysis
        # -------------------------------------------------

        binary_df = feature_df[
            feature_df["Normal Positive Rate"].notna()
        ].copy()

        if not binary_df.empty:

            print("\n")
            print(
                "BINARY FEATURE TARGET RATES"
            )

            print(
                binary_df[
                    [
                        "Feature",
                        "Normal Positive Rate",
                        "Suspicious Positive Rate",
                        "Positive Rate Difference"
                    ]
                ]
                .sort_values(
                    by="Positive Rate Difference",
                    key=lambda column:
                        column.abs(),
                    ascending=False
                )
                .head(15)
                .to_string(
                    index=False
                )
            )

        # -------------------------------------------------
        # Identify suspiciously weak features
        # -------------------------------------------------

        weak_auc_count = (
            feature_df[
                feature_df[
                    "Directional ROC-AUC"
                ] <= 0.55
            ]
            .shape[0]
        )

        useful_auc_count = (
            feature_df[
                feature_df[
                    "Directional ROC-AUC"
                ] > 0.60
            ]
            .shape[0]
        )

        # -------------------------------------------------
        # Save summary
        # -------------------------------------------------

        with open(
            TARGET_SUMMARY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "DATA QUALITY & TARGET VALIDATION SUMMARY\n"
            )

            file.write(
                "=" * 70
                + "\n\n"
            )

            file.write(
                f"Dataset Shape: {df.shape}\n"
            )

            file.write(
                f"Total Rows: {total_count}\n"
            )

            file.write(
                f"Total Features: {X_count(df, TARGET_COLUMN)}\n"
            )

            file.write(
                f"Normal Transactions: "
                f"{normal_count}\n"
            )

            file.write(
                f"Suspicious Transactions: "
                f"{suspicious_count}\n\n"
            )

            file.write(
                "Missing Values:\n"
            )

            if missing_values.empty:

                file.write(
                    "No missing values\n"
                )

            else:

                file.write(
                    missing_values.to_string()
                    + "\n"
                )

            file.write(
                "\nDuplicate Analysis:\n"
            )

            file.write(
                f"Exact Duplicate Rows: "
                f"{duplicate_rows}\n"
            )

            file.write(
                f"Repeated Feature Combinations: "
                f"{len(duplicate_groups)}\n"
            )

            file.write(
                f"Contradictory Feature Combinations: "
                f"{len(contradictory_groups)}\n"
            )

            file.write(
                "\nConstant Features:\n"
            )

            if constant_features:

                for feature in constant_features:

                    file.write(
                        f"- {feature}\n"
                    )

            else:

                file.write(
                    "None\n"
                )

            file.write(
                "\nIndividual Feature Separation:\n"
            )

            file.write(
                f"Features with Directional "
                f"ROC-AUC <= 0.55: "
                f"{weak_auc_count}\n"
            )

            file.write(
                f"Features with Directional "
                f"ROC-AUC > 0.60: "
                f"{useful_auc_count}\n"
            )

            file.write(
                "\nTop 15 Features:\n\n"
            )

            file.write(
                feature_df[
                    [
                        "Feature",
                        "Directional ROC-AUC",
                        "Normal Mean",
                        "Suspicious Mean",
                        "Standardized Difference"
                    ]
                ]
                .head(15)
                .to_string(
                    index=False
                )
            )

            file.write(
                "\n\nInterpretation:\n"
            )

            if useful_auc_count == 0:

                file.write(
                    "No individual feature has "
                    "Directional ROC-AUC above 0.60. "
                    "This indicates weak individual "
                    "predictive separation between "
                    "normal and suspicious transactions.\n"
                )

            else:

                file.write(
                    f"{useful_auc_count} feature(s) have "
                    "Directional ROC-AUC above 0.60. "
                    "These features may contain useful "
                    "individual predictive signal.\n"
                )

        # -------------------------------------------------
        # Completion
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print(
            "DATA QUALITY & TARGET VALIDATION COMPLETED"
        )
        print("=" * 70)

        print(
            f"\nExact duplicate rows: "
            f"{duplicate_rows}"
        )

        print(
            f"Contradictory feature combinations: "
            f"{len(contradictory_groups)}"
        )

        print(
            f"Constant features: "
            f"{len(constant_features)}"
        )

        print(
            f"Features with Directional "
            f"ROC-AUC > 0.60: "
            f"{useful_auc_count}"
        )

        print(
            f"Features with Directional "
            f"ROC-AUC <= 0.55: "
            f"{weak_auc_count}"
        )

        print("\nFiles saved:")

        print(
            DUPLICATE_FILE
        )

        print(
            FEATURE_LABEL_FILE
        )

        print(
            TARGET_SUMMARY_FILE
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


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def X_count(
    dataframe,
    target_column
):

    return len(
        dataframe.columns
    ) - 1


# ---------------------------------------------------------
# Run program
# ---------------------------------------------------------

if __name__ == "__main__":

    main()