import os
import sys
import joblib
import pandas as pd

# Project root directory
BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "final_model"))

from custom_transformers import (
    FeatureSelector,
    SelectiveScaler
)


# Input dataset
INPUT_FILE = os.path.join(
    BASE_DIR,
    "Dataset",
    "processed data",
    "encoded_transactions.csv"
)

# Trained model
MODEL_FILE = os.path.join(
    BASE_DIR,
    "final_model",
    "models",
    "ros_logistic_end_to_end_finetuned.pkl"
)

# Output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_model",
    "results"
)

# Output prediction file
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_fraud_predictions.csv"
)


# Optimized Decision Threshold (Max F2 / Recall-Weighted)
DECISION_THRESHOLD = 0.42


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
        # Load encoded dataset
        # ---------------------------------------------------------

        df = pd.read_csv(
            INPUT_FILE
        )

        print("Dataset loaded successfully.")

        print("\nDataset shape:")
        print(df.shape)

        # ---------------------------------------------------------
        # Check target column
        # ---------------------------------------------------------

        target_column = "Suspicious Activity Flag"

        if target_column not in df.columns:

            raise ValueError(
                f"Target column '{target_column}' was not found."
            )

        # ---------------------------------------------------------
        # Separate features
        #
        # The target column is removed because the model is used
        # to generate predictions from transaction features.
        # ---------------------------------------------------------

        X = df.drop(
            target_column,
            axis=1
        )

        # ---------------------------------------------------------
        # Load trained model pipeline
        # ---------------------------------------------------------

        print("\nLoading trained model...")

        model = joblib.load(
            MODEL_FILE
        )

        print("Model loaded successfully.")

        # ---------------------------------------------------------
        # Generate fraud probability & predictions (Optimized Threshold)
        #
        # Probability of class 1 = suspicious transaction
        # Decision cutoff: DECISION_THRESHOLD = 0.42
        # ---------------------------------------------------------

        print(f"\nGenerating predictions using optimized threshold ({DECISION_THRESHOLD:.2f})...")

        probabilities = model.predict_proba(
            X
        )[:, 1]

        predictions = (
            probabilities >= DECISION_THRESHOLD
        ).astype(int)

        # ---------------------------------------------------------
        # Create result dataframe
        # ---------------------------------------------------------

        results = df.copy()

        results["Predicted Class"] = predictions

        results["Fraud Probability"] = probabilities

        # ---------------------------------------------------------
        # Convert prediction into readable label
        # ---------------------------------------------------------

        results["Prediction"] = results[
            "Predicted Class"
        ].map({
            0: "Normal",
            1: "Suspicious"
        })

        # ---------------------------------------------------------
        # Create risk level
        #
        # This is only a probability-based display label.
        # It is not another machine-learning model.
        # ---------------------------------------------------------

        results["Risk Level"] = pd.cut(
            results["Fraud Probability"],
            bins=[
                -0.01,
                0.30,
                0.60,
                1.00
            ],
            labels=[
                "Low",
                "Medium",
                "High"
            ]
        )

        # ---------------------------------------------------------
        # Save prediction results
        # ---------------------------------------------------------

        results.to_csv(
            OUTPUT_FILE,
            index=False
        )

        # ---------------------------------------------------------
        # Display prediction summary
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("FINAL PREDICTION SUMMARY")
        print("=" * 60)

        total_transactions = len(
            results
        )

        normal_transactions = (
            results["Predicted Class"] == 0
        ).sum()

        suspicious_transactions = (
            results["Predicted Class"] == 1
        ).sum()

        print(
            f"Total Transactions:      {total_transactions}"
        )

        print(
            f"Normal Transactions:     {normal_transactions}"
        )

        print(
            f"Suspicious Transactions: {suspicious_transactions}"
        )

        suspicious_percentage = (
            suspicious_transactions
            / total_transactions
            * 100
        )

        print(
            f"Suspicious Percentage:    "
            f"{suspicious_percentage:.2f}%"
        )

        # ---------------------------------------------------------
        # Display risk-level summary
        # ---------------------------------------------------------

        print("\nRisk Level Summary:")
        print(
            results["Risk Level"]
            .value_counts()
        )

        # ---------------------------------------------------------
        # Display first 10 predictions
        # ---------------------------------------------------------

        print("\nFirst 10 Predictions:")

        print(
            results[
                [
                    "Predicted Class",
                    "Fraud Probability",
                    "Prediction",
                    "Risk Level"
                ]
            ].head(10)
        )

        # ---------------------------------------------------------
        # Display output file location
        # ---------------------------------------------------------

        print("\nFinal prediction file saved to:")
        print(OUTPUT_FILE)

        print(
            "\nFinal prediction completed successfully."
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