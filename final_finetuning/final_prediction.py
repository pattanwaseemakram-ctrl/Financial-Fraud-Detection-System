import os
import joblib
import pandas as pd


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
# Final trained model
# ---------------------------------------------------------

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "final_finetuning",
    "final_clean_model_results",
    "models",
    "final_clean_logistic_model.pkl"
)


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "final_finetuning",
    "final_prediction_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Output file
# ---------------------------------------------------------

PREDICTION_FILE = os.path.join(
    OUTPUT_DIR,
    "final_fraud_predictions.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "final_prediction_summary.txt"
)


# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

TARGET_COLUMN = "Suspicious Activity Flag"


# ---------------------------------------------------------
# Features removed during final model training
# ---------------------------------------------------------

REMOVED_FEATURES = [
    "Transaction_Month",
    "High_Value_Transaction",
    "Low_Balance",
    "High_Amount_Unusual_Hour",
    "Location_Deviation_High_Amount",
    "Amount_Percent_of_Balance"
]


# ---------------------------------------------------------
# Convert probability into risk level
# ---------------------------------------------------------

def assign_risk_level(probability):

    if probability < 0.30:

        return "Low"

    elif probability < 0.70:

        return "Medium"

    else:

        return "High"


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
        # Load final model
        # -------------------------------------------------

        print(
            "\nLoading final model..."
        )

        model = joblib.load(
            MODEL_FILE
        )

        print(
            "Final model loaded successfully."
        )

        # -------------------------------------------------
        # Remove target if present
        #
        # The prediction stage should not use the target.
        # -------------------------------------------------

        if TARGET_COLUMN in df.columns:

            X = df.drop(
                columns=[TARGET_COLUMN]
            ).copy()

        else:

            X = df.copy()

        # -------------------------------------------------
        # Remove the same features used during final
        # model training
        # -------------------------------------------------

        existing_removed_features = [
            feature
            for feature in REMOVED_FEATURES
            if feature in X.columns
        ]

        X_clean = X.drop(
            columns=existing_removed_features
        )

        print(
            f"\nPrediction feature shape: "
            f"{X_clean.shape}"
        )

        print(
            "\nFeatures removed:"
        )

        for feature in existing_removed_features:

            print(
                f"- {feature}"
            )

        # -------------------------------------------------
        # Generate predictions
        # -------------------------------------------------

        print(
            "\nGenerating predictions..."
        )

        predictions = model.predict(
            X_clean
        )

        probabilities = model.predict_proba(
            X_clean
        )[:, 1]

        # -------------------------------------------------
        # Create prediction DataFrame
        # -------------------------------------------------

        prediction_df = pd.DataFrame()

        # Row number
        prediction_df[
            "Transaction_Row"
        ] = range(
            1,
            len(df) + 1
        )

        # Prediction
        prediction_df[
            "Prediction"
        ] = predictions

        # Probability
        prediction_df[
            "Fraud_Probability"
        ] = probabilities

        # Percentage probability
        prediction_df[
            "Fraud_Probability_Percent"
        ] = (
            probabilities * 100
        )

        # Risk level
        prediction_df[
            "Risk_Level"
        ] = [
            assign_risk_level(
                probability
            )
            for probability in probabilities
        ]

        # -------------------------------------------------
        # Save predictions
        # -------------------------------------------------

        prediction_df.to_csv(
            PREDICTION_FILE,
            index=False
        )

        # -------------------------------------------------
        # Count predictions
        # -------------------------------------------------

        normal_count = (
            (predictions == 0)
            .sum()
        )

        suspicious_count = (
            (predictions == 1)
            .sum()
        )

        low_risk_count = (
            (
                prediction_df["Risk_Level"]
                == "Low"
            )
            .sum()
        )

        medium_risk_count = (
            (
                prediction_df["Risk_Level"]
                == "Medium"
            )
            .sum()
        )

        high_risk_count = (
            (
                prediction_df["Risk_Level"]
                == "High"
            )
            .sum()
        )

        # -------------------------------------------------
        # Display results
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL PREDICTION RESULTS")
        print("=" * 70)

        print(
            f"\nTotal transactions: "
            f"{len(predictions)}"
        )

        print(
            f"Predicted Normal: "
            f"{normal_count}"
        )

        print(
            f"Predicted Suspicious: "
            f"{suspicious_count}"
        )

        print(
            f"\nLow Risk: "
            f"{low_risk_count}"
        )

        print(
            f"Medium Risk: "
            f"{medium_risk_count}"
        )

        print(
            f"High Risk: "
            f"{high_risk_count}"
        )

        # -------------------------------------------------
        # Probability statistics
        # -------------------------------------------------

        minimum_probability = (
            probabilities.min()
        )

        maximum_probability = (
            probabilities.max()
        )

        average_probability = (
            probabilities.mean()
        )

        print(
            f"\nMinimum fraud probability: "
            f"{minimum_probability:.4f}"
        )

        print(
            f"Maximum fraud probability: "
            f"{maximum_probability:.4f}"
        )

        print(
            f"Average fraud probability: "
            f"{average_probability:.4f}"
        )

        # -------------------------------------------------
        # Display first 10 predictions
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FIRST 10 PREDICTIONS")
        print("=" * 70)

        print(
            prediction_df.head(10)
            .to_string(
                index=False
            )
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
                "FINAL FRAUD PREDICTION SUMMARY\n"
            )

            file.write(
                "=" * 70
                + "\n\n"
            )

            file.write(
                "Model: Final Clean Logistic Regression\n"
            )

            file.write(
                "Input Dataset: "
                "encoded_transactions_improved.csv\n\n"
            )

            file.write(
                f"Total Transactions: "
                f"{len(predictions)}\n"
            )

            file.write(
                f"Predicted Normal: "
                f"{normal_count}\n"
            )

            file.write(
                f"Predicted Suspicious: "
                f"{suspicious_count}\n\n"
            )

            file.write(
                "Risk Distribution:\n"
            )

            file.write(
                f"Low Risk: "
                f"{low_risk_count}\n"
            )

            file.write(
                f"Medium Risk: "
                f"{medium_risk_count}\n"
            )

            file.write(
                f"High Risk: "
                f"{high_risk_count}\n\n"
            )

            file.write(
                "Probability Statistics:\n"
            )

            file.write(
                f"Minimum: "
                f"{minimum_probability:.4f}\n"
            )

            file.write(
                f"Maximum: "
                f"{maximum_probability:.4f}\n"
            )

            file.write(
                f"Average: "
                f"{average_probability:.4f}\n\n"
            )

            file.write(
                "Removed Features:\n"
            )

            for feature in existing_removed_features:

                file.write(
                    f"- {feature}\n"
                )

        # -------------------------------------------------
        # Completion
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL PREDICTION COMPLETED")
        print("=" * 70)

        print(
            "\nPrediction file saved to:"
        )

        print(
            PREDICTION_FILE
        )

        print(
            "\nSummary saved to:"
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