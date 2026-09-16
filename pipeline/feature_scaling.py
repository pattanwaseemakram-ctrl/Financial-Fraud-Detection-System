import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Input and output file paths
INPUT_FILE = "Dataset/processed data/encoded_transactions.csv"

X_TRAIN_FILE = "Dataset/split/X_train.csv"
X_TEST_FILE = "Dataset/split/X_test.csv"

Y_TRAIN_FILE = "Dataset/split/y_train.csv"
Y_TEST_FILE = "Dataset/split/y_test.csv"


# Class containing feature scaling methods
class FeatureScaler:

    # Method to load the dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to display the first five rows
    def get_data_preview(self, df):
        return df.head()

    # Method to get dataset information
    def get_data_info(self, df):
        return df.info()

    # Method to separate features and target
    def separate_features_target(self, df):
        X = df.drop(
            "Suspicious Activity Flag",
            axis=1
        )

        y = df["Suspicious Activity Flag"]

        return X, y

    # Method to split the dataset into training and testing data
    def split_data(self, X, y):

        return train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

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

    # Method to fit scaler only on training data
    def fit_scaler(
        self,
        scaler,
        X_train,
        numerical_columns
    ):

        scaler.fit(
            X_train[numerical_columns]
        )

        return scaler

    # Method to transform training data
    def transform_training_data(
        self,
        scaler,
        X_train,
        numerical_columns
    ):

        X_train = X_train.copy()

        X_train[numerical_columns] = scaler.transform(
            X_train[numerical_columns]
        )

        return X_train

    # Method to transform testing data
    def transform_testing_data(
        self,
        scaler,
        X_test,
        numerical_columns
    ):

        X_test = X_test.copy()

        X_test[numerical_columns] = scaler.transform(
            X_test[numerical_columns]
        )

        return X_test

    # Method to get scaled training mean
    def get_scaled_mean(
        self,
        df,
        numerical_columns
    ):

        return df[numerical_columns].mean()

    # Method to get scaled training standard deviation
    def get_scaled_std(
        self,
        df,
        numerical_columns
    ):

        return df[numerical_columns].std()

    # Method to save training data
    def save_training_data(
        self,
        X_train,
        y_train
    ):

        X_train.to_csv(
            X_TRAIN_FILE,
            index=False
        )

        y_train.to_csv(
            Y_TRAIN_FILE,
            index=False
        )

        return True

    # Method to save testing data
    def save_testing_data(
        self,
        X_test,
        y_test
    ):

        X_test.to_csv(
            X_TEST_FILE,
            index=False
        )

        y_test.to_csv(
            Y_TEST_FILE,
            index=False
        )

        return True


# Main function to control the scaling workflow
def main():

    try:

        # Create FeatureScaler object
        scaler_processor = FeatureScaler()

        # Load encoded dataset
        df = scaler_processor.load_data(
            INPUT_FILE
        )

        print("\nFirst 5 Rows:")
        print(
            scaler_processor.get_data_preview(df)
        )

        print("\nDataset Information:")
        scaler_processor.get_data_info(df)

        # Separate features and target
        X, y = scaler_processor.separate_features_target(
            df
        )

        print("\nFeature shape:")
        print(X.shape)

        print("\nTarget distribution:")
        print(y.value_counts())

        # Split data before scaling
        X_train, X_test, y_train, y_test = (
            scaler_processor.split_data(X, y)
        )

        print("\nTraining data shape:")
        print(X_train.shape)

        print("Testing data shape:")
        print(X_test.shape)

        print("\nTraining target distribution:")
        print(y_train.value_counts())

        print("\nTesting target distribution:")
        print(y_test.value_counts())

        # Get numerical columns
        numerical_columns = (
            scaler_processor.get_numerical_columns()
        )

        print("\nColumns selected for scaling:")
        print(numerical_columns)

        # Create StandardScaler
        scaler = scaler_processor.create_scaler()

        print(
            "\nStandardScaler object created successfully."
        )

        # Fit the scaler ONLY on training data
        scaler = scaler_processor.fit_scaler(
            scaler,
            X_train,
            numerical_columns
        )

        print(
            "\nStandardScaler fitted only on training data."
        )

        # Transform training data
        X_train = (
            scaler_processor.transform_training_data(
                scaler,
                X_train,
                numerical_columns
            )
        )

        # Transform testing data using the same scaler
        X_test = (
            scaler_processor.transform_testing_data(
                scaler,
                X_test,
                numerical_columns
            )
        )

        print(
            "\nTraining and testing data transformed successfully."
        )

        # Display scaled training data
        print("\nScaled Training Data:")
        print(
            scaler_processor.get_data_preview(
                X_train
            )
        )

        # Display scaled test data
        print("\nScaled Testing Data:")
        print(
            scaler_processor.get_data_preview(
                X_test
            )
        )

        # Display training means
        print("\nTraining Scaled Means:")
        print(
            scaler_processor.get_scaled_mean(
                X_train,
                numerical_columns
            )
        )

        # Display training standard deviations
        print("\nTraining Scaled Standard Deviations:")
        print(
            scaler_processor.get_scaled_std(
                X_train,
                numerical_columns
            )
        )

        # Save training data
        train_saved = (
            scaler_processor.save_training_data(
                X_train,
                y_train
            )
        )

        # Save testing data
        test_saved = (
            scaler_processor.save_testing_data(
                X_test,
                y_test
            )
        )

        if train_saved and test_saved:

            print(
                "\nScaled train/test datasets saved successfully."
            )

            print(
                f"\nTraining features saved to: {X_TRAIN_FILE}"
            )

            print(
                f"Testing features saved to: {X_TEST_FILE}"
            )

            print(
                f"Training target saved to: {Y_TRAIN_FILE}"
            )

            print(
                f"Testing target saved to: {Y_TEST_FILE}"
            )

    # Handle missing input file
    except FileNotFoundError:

        print(
            "Error: Required input dataset was not found."
        )

    # Handle invalid values
    except ValueError as e:

        print(
            "Feature scaling data error:",
            e
        )

    # Handle unexpected errors
    except Exception as e:

        print(
            "Unexpected error during feature scaling:",
            e
        )


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()