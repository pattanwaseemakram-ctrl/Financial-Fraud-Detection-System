import pandas as pd
from sklearn.preprocessing import StandardScaler


# Input and output file paths
INPUT_FILE = "Dataset/processed data/encoded_transactions.csv"
OUTPUT_FILE = "Dataset/processed data/scaled_transactions.csv"


# Class containing all feature scaling methods
class FeatureScaler:

    # Method to load the dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to get the first five rows
    def get_data_preview(self, df):
        return df.head()

    # Method to get dataset information
    def get_data_info(self, df):
        return df.info()

    # Method to get column names
    def get_columns(self, df):
        return df.columns

    # Method to get columns selected for scaling
    def get_numerical_columns(self):
        numerical_columns = [
            "Amount",
            "Account Balance",
            "Transaction_Hour",
            "Transaction_Day",
            "Transaction_Month"
        ] 
        return numerical_columns

    # Method to create StandardScaler
    def create_scaler(self):
        return StandardScaler()

    # Method to apply StandardScaler
    def scale_features(self, df, numerical_columns, scaler):
        df[numerical_columns] = scaler.fit_transform(
            df[numerical_columns])
        return df

    # Method to get the mean of scaled columns
    def get_scaled_mean(self, df, numerical_columns):
        return df[numerical_columns].mean()

    # Method to get the standard deviation of scaled columns
    def get_scaled_std(self, df, numerical_columns):
        return df[numerical_columns].std()

    # Method to save the scaled dataset
    def save_scaled_data(self, df, file_path):
        df.to_csv(file_path, index=False)
        return True


# Main function to control the feature scaling workflow
def main():

    try:

        # Create a FeatureScaler object
        scaler_processor = FeatureScaler()

        # Load the encoded dataset
        df = scaler_processor.load_data(INPUT_FILE)

        # Display the first five rows
        print("\nFirst 5 Rows:")
        print(scaler_processor.get_data_preview(df))

        # Display dataset information
        print("\nDataset Information:")
        scaler_processor.get_data_info(df)

        # Display columns before scaling
        print("\nColumns Before Scaling:")
        print(scaler_processor.get_columns(df))

        # Get numerical columns selected for scaling
        numerical_columns = scaler_processor.get_numerical_columns()

        print("\nColumns Selected for Scaling:")
        print(numerical_columns)

        # Create StandardScaler object
        scaler = scaler_processor.create_scaler()

        print("\nStandardScaler object created successfully.")

        # Apply StandardScaler to selected columns
        df = scaler_processor.scale_features(
            df,numerical_columns,scaler)

        print("\nFeature Scaling Applied Successfully.")

        # Display scaled dataset
        print("\nScaled Dataset:")
        print(scaler_processor.get_data_preview(df))

        # Display mean of scaled columns
        print("\nMean of Scaled Columns:")
        print(
            scaler_processor.get_scaled_mean(
                df,
                numerical_columns))

        # Display standard deviation of scaled columns
        print("\nStandard Deviation of Scaled Columns:")
        print(
            scaler_processor.get_scaled_std(
                df,
                numerical_columns))

        # Save the scaled dataset
        save_status = scaler_processor.save_scaled_data(
            df,
            OUTPUT_FILE
        )

        if save_status:
            print("\nScaled dataset saved successfully.")

    # Handle missing input file
    except FileNotFoundError:
        print("Error: Required input dataset was not found.")

    # Handle invalid data values
    except ValueError as e:
        print("Feature scaling data error:", e)

    # Handle unexpected errors
    except Exception as e:
        print("Unexpected error during feature scaling:", e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()