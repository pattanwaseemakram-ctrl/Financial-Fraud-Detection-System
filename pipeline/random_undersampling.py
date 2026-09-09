import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


# Input and output file paths
X_TRAIN_FILE = "Dataset/split/X_train_selected.csv"
Y_TRAIN_FILE = "Dataset/split/y_train.csv"

X_TRAIN_RANDOM_UNDER_FILE = (
    "Dataset/random undersampling/X_train_random_under.csv"
)

Y_TRAIN_RANDOM_UNDER_FILE = (
    "Dataset/random undersampling/y_train_random_under.csv"
)


# Class containing all Random Undersampling processing methods
class RandomUndersamplingProcessor:

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

    # Method to create Random Undersampling object
    def create_random_undersampler(self):
        return RandomUnderSampler(
            sampling_strategy=0.5,
            random_state=42
        )

    # Method to apply Random Undersampling to training data
    def apply_random_undersampling(
        self,
        undersampler,
        X_train,
        y_train
    ):
        X_train_random_under, y_train_random_under = (
            undersampler.fit_resample(
                X_train,
                y_train
            )
        )

        return (
            X_train_random_under,
            y_train_random_under
        )

    # Method to save Random Undersampling training data
    def save_random_under_data(
        self,
        X_train_random_under,
        y_train_random_under
    ):

        # Create output folder if it does not exist
        os.makedirs(
            "Dataset/random undersampling",
            exist_ok=True
        )

        X_train_random_under.to_csv(
            X_TRAIN_RANDOM_UNDER_FILE,
            index=False
        )

        y_train_random_under.to_csv(
            Y_TRAIN_RANDOM_UNDER_FILE,
            index=False
        )

        return True


# Main function to control the Random Undersampling workflow
def main():

    try:

        # Create a RandomUndersamplingProcessor object
        processor = RandomUndersamplingProcessor()

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

        # Display class distribution before Random Undersampling
        print(
            "\nClass distribution BEFORE "
            "Random Undersampling:"
        )

        print(
            processor.get_class_distribution(y_train)
        )

        # Create Random Undersampling object
        undersampler = processor.create_random_undersampler()

        # Apply Random Undersampling to training data
        (
            X_train_random_under,
            y_train_random_under
        ) = processor.apply_random_undersampling(
            undersampler,
            X_train,
            y_train
        )

        # Display class distribution after Random Undersampling
        print(
            "\nClass distribution AFTER "
            "Random Undersampling:"
        )

        print(
            processor.get_class_distribution(
                y_train_random_under
            )
        )

        # Display shape before Random Undersampling
        print("\nShape BEFORE Random Undersampling:")

        print(
            processor.get_shape(X_train)
        )

        # Display shape after Random Undersampling
        print("\nShape AFTER Random Undersampling:")

        print(
            processor.get_shape(
                X_train_random_under
            )
        )

        # Save Random Undersampling training data
        save_status = processor.save_random_under_data(
            X_train_random_under,
            y_train_random_under
        )

        if save_status:
            print(
                "\nRandom Undersampling training "
                "data saved successfully."
            )

    # Handle missing training dataset
    except FileNotFoundError:
        print(
            "Error: Required training dataset "
            "file was not found."
        )

    # Handle invalid Random Undersampling data
    except ValueError as e:
        print(
            "Data error during Random Undersampling:",
            e
        )

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during "
            "Random Undersampling:",
            e
        )


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()