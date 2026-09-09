import pandas as pd


# Input and output file paths
INPUT_FILE = "Dataset/raw dataset/financial_transactions.csv"
OUTPUT_FILE = "Dataset/processed data/cleaned_transactions.csv"

# Class containing all data preprocessing methods
class DataPreprocessor:

    # Method to load the dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to check missing values
    def check_missing_values(self, df):
        return df.isnull().sum()

    # Method to check duplicate rows
    def check_duplicates(self, df):
        return df.duplicated().sum()

    # Method to get data types
    def display_data_types(self, df):
        return df.dtypes

    # Method to get numerical columns
    def display_numerical_columns(self, df):
        return df.select_dtypes(
            include=["int64", "float64"]
        ).columns

    # Method to get categorical columns
    def display_categorical_columns(self, df):
        return df.select_dtypes(
            include=["object"]
        ).columns

    # Method to calculate missing value percentage
    def check_missing_percentage(self, df):
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        return missing_percentage

    # Method to calculate duplicate percentage
    def check_duplicate_percentage(self, df):
        duplicate_percentage = (
            df.duplicated().sum() / len(df)
        ) * 100
        return duplicate_percentage

    # Method to get unique values
    def display_unique_values(self, df):
        unique_values = {
            "Type": df["Type"].unique(),
            "Location": df["Location"].unique()}
        return unique_values

    # Method to check negative values
    def check_negative_values(self, df):
        negative_values = {
            "Amount": (df["Amount"] < 0).sum(),
            "Account Balance": (df["Account Balance"] < 0).sum()}
        return negative_values

    # Method to convert Timestamp column
    def convert_timestamp(self, df):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df

    # Method to get Timestamp data type
    def get_timestamp_dtype(self, df):
        return df["Timestamp"].dtype

    # Method to save the processed dataset
    def save_cleaned_data(self, df, file_path):
        df.to_csv(file_path, index=False)
        return True


# Main function to control the preprocessing workflow
def main():

    try:

        # Create an object of DataPreprocessor class
        processor = DataPreprocessor()

        # Load the dataset
        df = processor.load_data(INPUT_FILE)

        # Check missing values
        print("\nMissing Values:")
        print(processor.check_missing_values(df))

        # Check duplicate rows
        print("\nDuplicate Rows:")
        print(processor.check_duplicates(df))

        # Display data types
        print("\nData Types:")
        print(processor.display_data_types(df))

        # Display numerical columns
        print("\nNumerical Columns:")
        print(processor.display_numerical_columns(df))

        # Display categorical columns
        print("\nCategorical Columns:")
        print(processor.display_categorical_columns(df))

        # Calculate missing value percentage
        print("\nMissing Value Percentage:")
        print(processor.check_missing_percentage(df))

        # Calculate duplicate percentage
        duplicate_percentage = processor.check_duplicate_percentage(df)
        print("\nDuplicate Percentage:")
        print(f"{duplicate_percentage:.2f}%")

        # Get unique values
        unique_values = processor.display_unique_values(df)

        print("\nUnique Values in Type:")
        print(unique_values["Type"])

        print("\nUnique Values in Location:")
        print(unique_values["Location"])

        # Check negative values
        negative_values = processor.check_negative_values(df)

        print("\nNegative Amount Values:")
        print(negative_values["Amount"])

        print("\nNegative Account Balance Values:")
        print(negative_values["Account Balance"])

        # Convert Timestamp column
        df = processor.convert_timestamp(df)

        # Display Timestamp data type after conversion
        print("\nTimestamp Data Type After Conversion:")
        print(processor.get_timestamp_dtype(df))

        # Save the processed dataset
        save_status = processor.save_cleaned_data(
            df,
            OUTPUT_FILE
        )

        if save_status:
            print("\nCleaned dataset saved successfully!")

        # Display successful completion message
        print("\nData preprocessing completed successfully.")

    # Handle missing input file
    except FileNotFoundError:
        print("Error: Input dataset was not found.")

    # Handle invalid data values
    except ValueError as e:
        print("Data processing error:", e)

    # Handle unexpected errors
    except Exception as e:
        print("Unexpected error during preprocessing:", e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()