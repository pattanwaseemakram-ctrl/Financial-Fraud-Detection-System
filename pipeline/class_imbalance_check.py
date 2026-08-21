import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Input dataset path
INPUT_FILE = "Dataset/encoded_transactions.csv"


# Target column
TARGET_COLUMN = "Suspicious Activity Flag"


# Class containing class imbalance checking methods
class ClassImbalanceChecker:

    # Method to load the dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to get the first five rows
    def get_data_preview(self, df):
        return df.head()

    # Method to get dataset shape
    def get_data_shape(self, df):
        return df.shape

    # Method to get dataset columns
    def get_columns(self, df):
        return df.columns

    # Method to get class distribution
    def get_class_distribution(self, df, target_column):
        return df[target_column].value_counts()

    # Method to get class distribution percentage
    def get_class_percentage(self, df, target_column):
        return (
            df[target_column].value_counts(normalize=True) * 100)

    # Method to display class distribution plot
    def display_class_distribution(
        self,
        df,
        target_column
    ):
        sns.countplot(
            x=target_column,
            data=df)

        plt.title("Class Distribution")
        plt.xlabel(target_column)
        plt.ylabel("Count")

        plt.show()

    # Method to get data types
    def get_data_types(self, df):
        return df.dtypes


# Main function to control the class imbalance workflow
def main():

    try:

        # Create a ClassImbalanceChecker object
        checker = ClassImbalanceChecker()

        # Load dataset
        df = checker.load_data(INPUT_FILE)

        # Display first five rows
        print("\nFirst 5 Rows:")
        print(checker.get_data_preview(df))

        # Display dataset shape
        print("\nDataset Shape:")
        print(checker.get_data_shape(df))

        # Display dataset columns
        print("\nDataset Columns:")
        print(checker.get_columns(df))

        # Display class distribution
        print("\nClass Distribution:")
        print(
            checker.get_class_distribution(
                df,
                TARGET_COLUMN))

        # Display class distribution percentage
        print("\nClass Distribution Percentage:")
        print(
            checker.get_class_percentage(
                df,
                TARGET_COLUMN))

        # Display class distribution plot
        checker.display_class_distribution(
            df,
            TARGET_COLUMN)

        # Display data types
        print("\nData Types:")
        print(checker.get_data_types(df))

        print("\nClass Imbalance Check Completed Successfully.")

    # Handle missing dataset
    except FileNotFoundError:
        print("Error: Dataset file was not found.")

    # Handle missing target column
    except KeyError as e:
        print(
            "Error: Target column was not found:",e)

    # Handle invalid data values
    except ValueError as e:
        print(
            "Class imbalance data error:",e)

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during class imbalance check:",e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()