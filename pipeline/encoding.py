import pandas as pd


# Input and output file paths
INPUT_FILE = "Dataset/processed data/feature_engineered_transactions.csv"
OUTPUT_FILE = "Dataset/processed data/encoded_transactions.csv"


# Class containing all encoding methods
class Encoder:

    # Method to load the dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to get the first five rows
    def get_data_preview(self, df):
        return df.head()

    # Method to get categorical columns
    def get_categorical_columns(self, df):
        return df.select_dtypes(include=["object"]).columns

    # Method to remove unnecessary columns
    def drop_unnecessary_columns(self, df):
        columns_to_drop = [
            "Transaction ID",
            "Sender Account ID",
            "Receiver Account ID",
            "Timestamp",
            "Device ID",
            "IP Address"
        ]

        df = df.drop(columns=columns_to_drop)
        return df

    # Method to get unique values of categorical columns
    def get_unique_values(self, df):
        unique_values = {
            "Type": df["Type"].unique(),
            "Location": df["Location"].unique(),
            "Transaction_Weekday": df["Transaction_Weekday"].unique()}
        return unique_values

    # Method to perform One-Hot Encoding
    def one_hot_encode(self, df):
        df = pd.get_dummies(
            df,
            columns=[
                "Type",
                "Location",
                "Transaction_Weekday"],
            dtype=int)
        return df

    # Method to get encoded dataset preview
    def get_encoded_preview(self, df):
        return df.head()

    # Method to get encoded columns
    def get_encoded_columns(self, df):
        return df.columns

    # Method to save the encoded dataset
    def save_encoded_data(self, df, file_path):
        df.to_csv(file_path, index=False)
        return True


# Main function to control the encoding workflow
def main():

    try:

        # Create an Encoder object
        encoder = Encoder()

        # Load the feature-engineered dataset
        df = encoder.load_data(INPUT_FILE)

        # Display the first five rows
        print("\nFirst 5 Rows:")
        print(encoder.get_data_preview(df))

        # Display categorical columns
        print("\nCategorical Columns:")
        print(encoder.get_categorical_columns(df))

        # Drop unnecessary columns
        df = encoder.drop_unnecessary_columns(df)

        # Display remaining columns
        print("\nRemaining Columns:")
        print(df.columns)

        # Get unique values from categorical columns
        unique_values = encoder.get_unique_values(df)

        print("\nType:")
        print(unique_values["Type"])

        print("\nLocation:")
        print(unique_values["Location"])

        print("\nTransaction Weekday:")
        print(unique_values["Transaction_Weekday"])

        # Perform One-Hot Encoding
        df = encoder.one_hot_encode(df)

        # Display encoded dataset
        print("\nEncoded Dataset:")
        print(encoder.get_encoded_preview(df))

        # Display encoded columns
        print("\nEncoded Columns:")
        print(encoder.get_encoded_columns(df))

        # Save the encoded dataset
        save_status = encoder.save_encoded_data(
            df,
            OUTPUT_FILE
        )

        if save_status:
            print("\nEncoded dataset saved successfully.")

    # Handle missing input file
    except FileNotFoundError:
        print("Error: Required input dataset was not found.")

    # Handle invalid data values
    except ValueError as e:
        print("Encoding data error:", e)

    # Handle unexpected errors
    except Exception as e:
        print("Unexpected error during encoding:", e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()