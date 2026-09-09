import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Input dataset path
INPUT_FILE = "Dataset/processed data/feature_engineered_transactions.csv"


# Class containing all outlier detection methods
class OutlierDetector:

    # Method to load the dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to get numerical columns
    def get_numerical_columns(self, df):
        return df.select_dtypes(
            include=["int64", "float64"]
        ).columns

    # Method to display Amount box plot
    def display_amount_boxplot(self, df):
        plt.figure(figsize=(8, 5))

        sns.boxplot(x=df["Amount"])

        plt.title("Box Plot of Transaction Amount")
        plt.xlabel("Amount")

        plt.show()

    # Method to calculate IQR limits for Amount
    def calculate_amount_iqr(self, df):

        Q1 = df["Amount"].quantile(0.25)
        Q3 = df["Amount"].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - (1.5 * IQR)
        upper_limit = Q3 + (1.5 * IQR)

        return Q1, Q3, IQR, lower_limit, upper_limit

    # Method to detect Amount outliers
    def detect_amount_outliers(
        self,
        df,
        lower_limit,
        upper_limit
    ):
        outliers = df[
            (df["Amount"] < lower_limit) |
            (df["Amount"] > upper_limit)
        ]
        return outliers

    # Method to display Account Balance box plot
    def display_balance_boxplot(self, df):
        plt.figure(figsize=(8, 5))

        sns.boxplot(x=df["Account Balance"])

        plt.title("Box Plot of Account Balance")
        plt.xlabel("Account Balance")

        plt.show()

    # Method to calculate IQR limits for Account Balance
    def calculate_balance_iqr(self, df):

        Q1_balance = df["Account Balance"].quantile(0.25)
        Q3_balance = df["Account Balance"].quantile(0.75)

        IQR_balance = Q3_balance - Q1_balance

        lower_limit_balance = (
            Q1_balance - (1.5 * IQR_balance)
        )

        upper_limit_balance = (
            Q3_balance + (1.5 * IQR_balance)
        )

        return (
            Q1_balance,
            Q3_balance,
            IQR_balance,
            lower_limit_balance,
            upper_limit_balance)

    # Method to detect Account Balance outliers
    def detect_balance_outliers(
        self,
        df,
        lower_limit_balance,
        upper_limit_balance
    ):
        outliers_balance = df[
            (df["Account Balance"] < lower_limit_balance) |
            (df["Account Balance"] > upper_limit_balance)]
        return outliers_balance


# Main function to control the outlier detection workflow
def main():

    try:

        # Create an OutlierDetector object
        detector = OutlierDetector()

        # Load dataset
        df = detector.load_data(INPUT_FILE)

        # Display numerical columns
        print("\nNumerical Columns:")
        print(detector.get_numerical_columns(df))

        # Display Amount box plot
        detector.display_amount_boxplot(df)

        # Calculate IQR for Amount
        (
            Q1,
            Q3,
            IQR,
            lower_limit,
            upper_limit
        ) = detector.calculate_amount_iqr(df)

        # Display Amount IQR values
        print("\n---------- Transaction Amount ----------")
        print("Q1:", Q1)
        print("Q3:", Q3)
        print("IQR:", IQR)
        print("Lower Limit:", lower_limit)
        print("Upper Limit:", upper_limit)

        # Detect Amount outliers
        outliers = detector.detect_amount_outliers(
            df,
            lower_limit,
            upper_limit
        )

        # Display Amount outlier results
        print("\nNumber of Outliers:", len(outliers))
        print("\nFirst 5 Amount Outliers:")
        print(outliers.head())

        # Display Account Balance box plot
        detector.display_balance_boxplot(df)

        # Calculate IQR for Account Balance
        (
            Q1_balance,
            Q3_balance,
            IQR_balance,
            lower_limit_balance,
            upper_limit_balance
        ) = detector.calculate_balance_iqr(df)

        # Display Account Balance IQR values
        print("\n---------- Account Balance ----------")
        print("Q1:", Q1_balance)
        print("Q3:", Q3_balance)
        print("IQR:", IQR_balance)
        print("Lower Limit:", lower_limit_balance)
        print("Upper Limit:", upper_limit_balance)

        # Detect Account Balance outliers
        outliers_balance = detector.detect_balance_outliers(
            df,
            lower_limit_balance,
            upper_limit_balance
        )

        # Display Account Balance outlier results
        print(
            "\nNumber of Account Balance Outliers:",
            len(outliers_balance))

        print("\nFirst 5 Account Balance Outliers:")
        print(outliers_balance.head())

        print("\nOutlier Detection Completed Successfully.")

    # Handle missing dataset
    except FileNotFoundError:
        print("Error: Dataset file was not found.")

    # Handle missing columns
    except KeyError as e:
        print(
            "Error: Required column was not found:",e)

    # Handle invalid data values
    except ValueError as e:
        print(
            "Outlier detection data error:",e)

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during outlier detection:",e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()