import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Input dataset
# ---------------------------------------------------------

INPUT_FILE = (
    "Dataset/processed data/encoded_transactions_improved.csv"
)

# Output directory
OUTPUT_DIR = "feature_analysis/results"


# Output files
IMPORTANCE_FILE = (
    "feature_analysis/results/"
    "random_forest_feature_importance.csv"
)

MUTUAL_INFO_FILE = (
    "feature_analysis/results/"
    "mutual_information_scores.csv"
)

CLASS_STATS_FILE = (
    "feature_analysis/results/"
    "class_wise_feature_statistics.csv"
)

SUMMARY_FILE = (
    "feature_analysis/results/"
    "feature_signal_summary.txt"
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
        # Load improved encoded dataset
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
        # Validate target column
        # ---------------------------------------------------------

        if TARGET_COLUMN not in df.columns:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' was not found."
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
        # Create train/test split
        #
        # Feature importance and mutual information are calculated
        # using training data only.
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
        # Random Forest Feature Importance
        # ---------------------------------------------------------

        print("\nCalculating Random Forest feature importance...")

        random_forest = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

        random_forest.fit(
            X_train,
            y_train
        )

        feature_importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": random_forest.feature_importances_
        })

        feature_importance = (
            feature_importance
            .sort_values(
                by="Importance",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        # ---------------------------------------------------------
        # Mutual Information
        # ---------------------------------------------------------

        print(
            "\nCalculating mutual information scores..."
        )

        mutual_information = mutual_info_classif(
            X_train,
            y_train,
            random_state=42
        )

        mutual_info_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Mutual Information": mutual_information
        })

        mutual_info_df = (
            mutual_info_df
            .sort_values(
                by="Mutual Information",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        # ---------------------------------------------------------
        # Class-wise statistics
        # ---------------------------------------------------------

        print(
            "\nCalculating class-wise feature statistics..."
        )

        normal_data = X_train[
            y_train == 0
        ]

        suspicious_data = X_train[
            y_train == 1
        ]

        statistics = []

        for feature in X_train.columns:

            normal_mean = normal_data[
                feature
            ].mean()

            suspicious_mean = suspicious_data[
                feature
            ].mean()

            normal_median = normal_data[
                feature
            ].median()

            suspicious_median = suspicious_data[
                feature
            ].median()

            mean_difference = (
                suspicious_mean
                - normal_mean
            )

            absolute_mean_difference = abs(
                mean_difference
            )

            statistics.append({
                "Feature": feature,
                "Normal Mean": normal_mean,
                "Suspicious Mean": suspicious_mean,
                "Normal Median": normal_median,
                "Suspicious Median": suspicious_median,
                "Mean Difference": mean_difference,
                "Absolute Mean Difference":
                    absolute_mean_difference
            })

        class_stats_df = pd.DataFrame(
            statistics
        )

        class_stats_df = (
            class_stats_df
            .sort_values(
                by="Absolute Mean Difference",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        # ---------------------------------------------------------
        # Combine all feature signals
        # ---------------------------------------------------------

        signal_summary = feature_importance.merge(
            mutual_info_df,
            on="Feature",
            how="outer"
        )

        signal_summary = signal_summary.merge(
            class_stats_df[
                [
                    "Feature",
                    "Normal Mean",
                    "Suspicious Mean",
                    "Mean Difference",
                    "Absolute Mean Difference"
                ]
            ],
            on="Feature",
            how="left"
        )

        # ---------------------------------------------------------
        # Add ranking columns
        # ---------------------------------------------------------

        signal_summary["RF Rank"] = (
            signal_summary[
                "Importance"
            ]
            .rank(
                ascending=False,
                method="min"
            )
        )

        signal_summary["MI Rank"] = (
            signal_summary[
                "Mutual Information"
            ]
            .rank(
                ascending=False,
                method="min"
            )
        )

        signal_summary["Mean Difference Rank"] = (
            signal_summary[
                "Absolute Mean Difference"
            ]
            .rank(
                ascending=False,
                method="min"
            )
        )

        # ---------------------------------------------------------
        # Calculate overall signal rank
        # ---------------------------------------------------------

        signal_summary["Average Rank"] = (
            signal_summary["RF Rank"]
            + signal_summary["MI Rank"]
            + signal_summary["Mean Difference Rank"]
        ) / 3

        signal_summary = (
            signal_summary
            .sort_values(
                by="Average Rank",
                ascending=True
            )
            .reset_index(
                drop=True
            )
        )

        # ---------------------------------------------------------
        # Save individual analysis files
        # ---------------------------------------------------------

        feature_importance.to_csv(
            IMPORTANCE_FILE,
            index=False
        )

        mutual_info_df.to_csv(
            MUTUAL_INFO_FILE,
            index=False
        )

        class_stats_df.to_csv(
            CLASS_STATS_FILE,
            index=False
        )

        # ---------------------------------------------------------
        # Save combined summary
        # ---------------------------------------------------------

        signal_summary.to_csv(
            SUMMARY_FILE.replace(
                ".txt",
                ".csv"
            ),
            index=False
        )

        # ---------------------------------------------------------
        # Display Random Forest importance
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("TOP RANDOM FOREST FEATURES")
        print("=" * 60)

        print(
            feature_importance.head(
                15
            ).to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Display Mutual Information
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("TOP MUTUAL INFORMATION FEATURES")
        print("=" * 60)

        print(
            mutual_info_df.head(
                15
            ).to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Display class-wise differences
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("TOP CLASS-WISE DIFFERENCES")
        print("=" * 60)

        print(
            class_stats_df.head(
                15
            ).to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Display overall signal ranking
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("OVERALL FEATURE SIGNAL RANKING")
        print("=" * 60)

        print(
            signal_summary[
                [
                    "Feature",
                    "Importance",
                    "Mutual Information",
                    "Absolute Mean Difference",
                    "Average Rank"
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # Write text summary
        # ---------------------------------------------------------

        with open(
            SUMMARY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "FEATURE SIGNAL ANALYSIS SUMMARY\n"
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
                f"Training Samples: {len(X_train)}\n"
            )

            file.write(
                f"Testing Samples: {len(X_test)}\n\n"
            )

            file.write(
                "Top 15 Features by Overall Signal Ranking:\n\n"
            )

            file.write(
                signal_summary[
                    [
                        "Feature",
                        "Importance",
                        "Mutual Information",
                        "Absolute Mean Difference",
                        "Average Rank"
                    ]
                ]
                .head(15)
                .to_string(
                    index=False
                )
            )

            file.write(
                "\n\nTop Random Forest Features:\n\n"
            )

            file.write(
                feature_importance
                .head(15)
                .to_string(
                    index=False
                )
            )

            file.write(
                "\n\nTop Mutual Information Features:\n\n"
            )

            file.write(
                mutual_info_df
                .head(15)
                .to_string(
                    index=False
                )
            )

            file.write(
                "\n\nTop Class-wise Feature Differences:\n\n"
            )

            file.write(
                class_stats_df
                .head(15)
                .to_string(
                    index=False
                )
            )

        # ---------------------------------------------------------
        # Display output files
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("FEATURE SIGNAL ANALYSIS COMPLETED")
        print("=" * 60)

        print("\nFiles saved:")

        print(
            IMPORTANCE_FILE
        )

        print(
            MUTUAL_INFO_FILE
        )

        print(
            CLASS_STATS_FILE
        )

        print(
            SUMMARY_FILE
        )

        print(
            SUMMARY_FILE.replace(
                ".txt",
                ".csv"
            )
        )

        print(
            "\nFeature signal analysis completed successfully."
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