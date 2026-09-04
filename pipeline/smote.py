import pandas as pd
from imblearn.over_sampling import SMOTE


# Input and output file paths
X_TRAIN_FILE = "Dataset/split/X_train_selected.csv"
Y_TRAIN_FILE = "Dataset/split/y_train.csv"

X_TRAIN_SMOTE_FILE = "Dataset/smote/X_train_smote.csv"
Y_TRAIN_SMOTE_FILE = "Dataset/smote/y_train_smote.csv"


# Class containing all SMOTE processing methods
class SMOTEProcessor:

    # Method to load training features
    def load_features(self, file_path):
        return pd.read_csv(file_path)

    # Method to load training target
    def load_target(self, file_path):
        return pd.read_csv(file_path).squeeze()

    # Method to get dataset shape
    def get_shape(self, data):
        return data.shape

    # Method to get class distribution
    def get_class_distribution(self, y):
        return y.value_counts()

    # Method to create SMOTE object
    def create_smote(self):
        return SMOTE(sampling_strategy=0.5,random_state=42)

    # Method to apply SMOTE to training data
    def apply_smote(self, smote, X_train, y_train):
        X_train_smote, y_train_smote = smote.fit_resample(
            X_train,
            y_train)
        return X_train_smote, y_train_smote

    # Method to save SMOTE-balanced training data
    def save_smote_data(
        self,
        X_train_smote,
        y_train_smote
    ):
        X_train_smote.to_csv(
            X_TRAIN_SMOTE_FILE,
            index=False)

        y_train_smote.to_csv(
            Y_TRAIN_SMOTE_FILE,
            index=False)
        return True


# Main function to control the SMOTE workflow
def main():

    try:

        # Create a SMOTEProcessor object
        processor = SMOTEProcessor()

        # Load training features
        X_train = processor.load_features(
            X_TRAIN_FILE
        )

        # Load training target
        y_train = processor.load_target(
            Y_TRAIN_FILE
        )

        # Display training data shapes
        print(
            "Training features shape:",
            processor.get_shape(X_train))

        print(
            "Training target shape:",
            processor.get_shape(y_train))

        # Display class distribution before SMOTE
        print("\nClass distribution BEFORE SMOTE:")
        print(processor.get_class_distribution(y_train))

        # Create SMOTE object
        smote = processor.create_smote()

        # Apply SMOTE to training data
        X_train_smote, y_train_smote = processor.apply_smote(
            smote,
            X_train,
            y_train)

        # Display class distribution after SMOTE
        print("\nClass distribution AFTER SMOTE:")
        print(
            processor.get_class_distribution(
                y_train_smote))

        # Display shape before SMOTE
        print("\nShape BEFORE SMOTE:")
        print(processor.get_shape(X_train))

        # Display shape after SMOTE
        print("\nShape AFTER SMOTE:")
        print(processor.get_shape(X_train_smote))

        # Save SMOTE-balanced training data
        save_status = processor.save_smote_data(
            X_train_smote,
            y_train_smote)

        if save_status:
            print("\nSMOTE training data saved successfully.")

    # Handle missing training dataset
    except FileNotFoundError:
        print(
            "Error: Required training dataset file was not found.")

    # Handle invalid SMOTE data
    except ValueError as e:
        print(
            "Data error during SMOTE:",e)

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during SMOTE:",e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()