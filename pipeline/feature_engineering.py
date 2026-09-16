import pandas as pd


# Input and output file paths
INPUT_FILE = "Dataset/processed data/cleaned_transactions.csv"
OUTPUT_FILE = "Dataset/processed data/feature_engineered_transactions.csv"


# Class containing feature engineering methods
class FeatureEngineering:

    # Method to load the dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to convert Timestamp column
    def convert_timestamp(self, df):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df

    # Method to create time-based features
    def create_time_features(self, df):

        df["Transaction_Hour"] = df["Timestamp"].dt.hour
        df["Transaction_Day"] = df["Timestamp"].dt.day
        df["Transaction_Month"] = df["Timestamp"].dt.month
        df["Transaction_Weekday"] = df["Timestamp"].dt.day_name()

        return df

    # Method to create transaction and balance features
    def create_transaction_features(self, df):

        # Existing feature:
        # Identify high-value transactions
        df["High_Value_Transaction"] = (
            df["Amount"] > 10000
        ).astype(int)

        # Existing feature:
        # Identify transactions made when account balance is low
        df["Low_Balance"] = (
            df["Account Balance"] < 1000
        ).astype(int)

        # ---------------------------------------------------------
        # New feature:
        # Ratio of transaction amount to account balance
        #
        # This represents how large the transaction is compared
        # with the available account balance.
        # ---------------------------------------------------------

        df["Amount_to_Balance_Ratio"] = (
            df["Amount"] /
            df["Account Balance"].replace(0, 1)
        )

        # ---------------------------------------------------------
        # New feature:
        # Transaction amount as a percentage of account balance
        # ---------------------------------------------------------

        df["Amount_Percent_of_Balance"] = (
            df["Amount_to_Balance_Ratio"] * 100
        )

        # ---------------------------------------------------------
        # New feature:
        # Interaction between transaction amount and spending
        # pattern deviation
        #
        # Higher values indicate large transactions combined
        # with stronger deviation from normal spending behavior.
        # ---------------------------------------------------------

        df["Spending_Deviation_Amount"] = (
            df["Amount"] *
            df["Spending Pattern Deviation"]
        )

        # ---------------------------------------------------------
        # New feature:
        # Location deviation
        #
        # Login Location Match:
        # 1 = location matches
        # 0 = location does not match
        #
        # Therefore:
        # 1 - Login Location Match
        # gives:
        # 0 = location matches
        # 1 = location mismatch
        # ---------------------------------------------------------

        df["Location_Deviation"] = (
            1 -
            df["Login Location Match"]
        )

        # ---------------------------------------------------------
        # New feature:
        # Identify transactions made during unusual hours
        #
        # Here we consider:
        # 00:00-05:00 and 22:00-23:00
        # as unusual transaction hours.
        # ---------------------------------------------------------

        df["Unusual_Hour"] = (
            (
                (df["Transaction_Hour"] < 6) |
                (df["Transaction_Hour"] >= 22)
            )
        ).astype(int)

        # ---------------------------------------------------------
        # New feature:
        # Combination of high transaction value and unusual hour
        # ---------------------------------------------------------

        df["High_Amount_Unusual_Hour"] = (
            df["High_Value_Transaction"] *
            df["Unusual_Hour"]
        )

        # ---------------------------------------------------------
        # New feature:
        # Combination of location mismatch and high transaction
        # value
        # ---------------------------------------------------------

        df["Location_Deviation_High_Amount"] = (
            df["Location_Deviation"] *
            df["High_Value_Transaction"]
        )

        return df

    # Method to get the first five rows
    def get_data_preview(self, df):
        return df.head()

    # Method to save the engineered dataset
    def save_data(self, df, file_path):
        df.to_csv(file_path, index=False)
        return True


# Main function
def main():

    try:

        # Create FeatureEngineering object
        feature_engineering = FeatureEngineering()

        # Load dataset
        df = feature_engineering.load_data(INPUT_FILE)

        # Convert Timestamp
        df = feature_engineering.convert_timestamp(df)

        # Create time-based features
        df = feature_engineering.create_time_features(df)

        # Create transaction and balance features
        df = feature_engineering.create_transaction_features(df)

        # Get dataset preview
        preview = feature_engineering.get_data_preview(df)
        print(preview)

        # Save engineered dataset
        save_status = feature_engineering.save_data(
            df,
            OUTPUT_FILE
        )

        if save_status:
            print(
                "Feature Engineering Completed Successfully."
            )

    # Handle missing input file
    except FileNotFoundError:
        print(
            "Error: Required input dataset was not found."
        )

    # Handle invalid data values
    except ValueError as e:
        print(
            "Feature engineering data error:",
            e
        )

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during feature engineering:",
            e
        )


# Run main() when the file is executed directly
if __name__ == "__main__":
    main()