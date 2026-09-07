import os
import pandas as pd
from imblearn.over_sampling import ADASYN


# Input and output file paths
X_TRAIN_FILE = "Dataset/split/X_train_selected.csv"
Y_TRAIN_FILE = "Dataset/split/y_train.csv"

X_TRAIN_ADASYN_FILE = (
    "Dataset/adasyn/X_train_adasyn.csv"
)

Y_TRAIN_ADASYN_FILE = (
    "Dataset/adasyn/y_train_adasyn.csv"
)


# Class containing all ADASYN processing methods
class ADASYNProcessor:

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

    # Method to create ADASYN object
    def create_adasyn(self):
        return ADASYN(
            sampling_strategy=0.5,
            random_state=42
        )

    # Method to apply ADASYN to training data
    def apply_adasyn(
        self,
        adasyn,
        X_train,
        y_train
    ):
        X_train_adasyn, y_train_adasyn = (
            adasyn.fit_resample(
                X_train,
                y_train
            )
        )

        return (
            X_train_adasyn,
            y_train_adasyn
        )

    # Method to save ADASYN-balanced training data
    def save_adasyn_data(
        self,
        X_train_adasyn,
        y_train_adasyn
    ):

        # Create output folder if it does not exist
        os.makedirs(
            "Dataset/adasyn",
            exist_ok=True
        )

        X_train_adasyn.to_csv(
            X_TRAIN_ADASYN_FILE,
            index=False
        )

        y_train_adasyn.to_csv(
            Y_TRAIN_ADASYN_FILE,
            index=False
        )

        return True


# Main function to control the ADASYN workflow
def main():

    try:

        # Create an ADASYNProcessor object
        processor = ADASYNProcessor()

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
            processor.get_shape(X_train)
        )

        print(
            "Training target shape:",
            processor.get_shape(y_train)
        )

        # Display class distribution before ADASYN
        print(
            "\nClass distribution BEFORE ADASYN:"
        )

        print(
            processor.get_class_distribution(y_train)
        )

        # Create ADASYN object
        adasyn = processor.create_adasyn()

        # Apply ADASYN to training data
        (
            X_train_adasyn,
            y_train_adasyn
        ) = processor.apply_adasyn(
            adasyn,
            X_train,
            y_train
        )

        # Display class distribution after ADASYN
        print(
            "\nClass distribution AFTER ADASYN:"
        )

        print(
            processor.get_class_distribution(
                y_train_adasyn
            )
        )

        # Display shape before ADASYN
        print("\nShape BEFORE ADASYN:")

        print(
            processor.get_shape(X_train)
        )

        # Display shape after ADASYN
        print("\nShape AFTER ADASYN:")

        print(
            processor.get_shape(
                X_train_adasyn
            )
        )

        # Save ADASYN-balanced training data
        save_status = processor.save_adasyn_data(
            X_train_adasyn,
            y_train_adasyn
        )

        if save_status:
            print(
                "\nADASYN training data "
                "saved successfully."
            )

    # Handle missing training dataset
    except FileNotFoundError:
        print(
            "Error: Required training dataset "
            "file was not found."
        )

    # Handle invalid ADASYN data
    except ValueError as e:
        print(
            "Data error during ADASYN:",
            e
        )

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during ADASYN:",
            e
        )


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()