import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Input dataset path
INPUT_FILE = "Dataset/raw dataset/financial_transactions.csv"


# EDA class containing data analysis methods
class EDA:

    # Load the dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Check for missing values
    def check_missing_values(self, df):
        return df.isnull().sum()

    # Check for duplicate rows
    def check_duplicates(self, df):
        return df.duplicated().sum()

    # Get the first five records
    def display_first_rows(self, df):
        return df.head()

    # Get the last five records
    def display_last_rows(self, df):
        return df.tail()

    # Get dataset shape
    def display_shape(self, df):
        return df.shape

    # Get column names
    def display_columns(self, df):
        return df.columns

    # Get dataset information
    def display_info(self, df):
        return df.info()

    # Get data types
    def display_data_types(self, df):
        return df.dtypes

    # Get descriptive statistics
    def display_statistics(self, df):
        return df.describe()

    # Get unique values of categorical columns
    def display_unique_values(self, df):
        unique_values = {}

        for col in df.select_dtypes(include="object").columns:
            unique_values[col] = df[col].unique()
            return unique_values

    # Get frequency of Type and Location values
    def display_value_counts(self, df):
        value_counts = {
            "Type": df["Type"].value_counts(),
            "Location": df["Location"].value_counts()}
        return value_counts



# Main function to execute the EDA workflow
def main():

    try:

        # Create an object of the EDA class
        eda = EDA()

        # Load the dataset
        df = eda.load_data(INPUT_FILE)

        # Perform basic EDA operations
        eda.check_missing_values(df)
        eda.check_duplicates(df)
        eda.display_first_rows(df)
        eda.display_last_rows(df)
        eda.display_shape(df)
        eda.display_columns(df)
        eda.display_info(df)
        eda.display_data_types(df)
        eda.display_statistics(df)
        eda.display_unique_values(df)
        eda.display_value_counts(df)

        # Display successful completion message
        print("\nEDA completed successfully.")

    # Handle missing input file
    except FileNotFoundError:
        print("Error: EDA input dataset was not found.")

    # Handle invalid data values
    except ValueError as e:
        print("EDA data error:", e)

    # Handle unexpected errors
    except Exception as e:
        print("Unexpected error during EDA:", e)


# Run the main function when the file is executed directly
if __name__ == "__main__":
    main()