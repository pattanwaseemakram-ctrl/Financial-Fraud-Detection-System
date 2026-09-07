import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


# Input and output file paths
X_TRAIN_FILE = "Dataset/split/X_train_selected.csv"
Y_TRAIN_FILE = "Dataset/split/y_train.csv"

X_TRAIN_RANDOM_OVER_FILE = (
    "Dataset/random oversampling/X_train_random_over.csv"
)

Y_TRAIN_RANDOM_OVER_FILE = (
    "Dataset/random oversampling/y_train_random_over.csv"
)


# Class containing all Random Oversampling processing methods
class RandomOversamplingProcessor:

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

    # Method to create Random Oversampling object
    def create_random_oversampler(self):
        return RandomOverSampler(
            sampling_strategy=0.5,
            random_state=42
        )

    # Method to apply Random Oversampling to training data
    def apply_random_oversampling(
        self,
        oversampler,
        X_train,
        y_train
    ):
        X_train_random_over, y_train_random_over = (
            oversampler.fit_resample(
                X_train,
                y_train
            )
        )

        return (
            X_train_random_over,
            y_train_random_over
        )

    # Method to save Random Oversampling training data
    def save_random_over_data(
        self,
        X_train_random_over,
        y_train_random_over
    ):

        # Create output folder if it does not exist
        os.makedirs(
            "Dataset/random oversampling",
            exist_ok=True
        )

        X_train_random_over.to_csv(
            X_TRAIN_RANDOM_OVER_FILE,
            index=False
        )

        y_train_random_over.to_csv(
            Y_TRAIN_RANDOM_OVER_FILE,
            index=False
        )

        return True


# Main function to control the Random Oversampling workflow
def main():

    try:

        # Create a RandomOversamplingProcessor object
        processor = RandomOversamplingProcessor()

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

        # Display class distribution before Random Oversampling
        print(
            "\nClass distribution BEFORE "
            "Random Oversampling:"
        )

        print(
            processor.get_class_distribution(y_train)
        )

        # Create Random Oversampling object
        oversampler = processor.create_random_oversampler()

        # Apply Random Oversampling to training data
        (
            X_train_random_over,
            y_train_random_over
        ) = processor.apply_random_oversampling(
            oversampler,
            X_train,
            y_train
        )

        # Display class distribution after Random Oversampling
        print(
            "\nClass distribution AFTER "
            "Random Oversampling:"
        )

        print(
            processor.get_class_distribution(
                y_train_random_over
            )
        )

        # Display shape before Random Oversampling
        print("\nShape BEFORE Random Oversampling:")

        print(
            processor.get_shape(X_train)
        )

        # Display shape after Random Oversampling
        print("\nShape AFTER Random Oversampling:")

        print(
            processor.get_shape(
                X_train_random_over
            )
        )

        # Save Random Oversampling training data
        save_status = processor.save_random_over_data(
            X_train_random_over,
            y_train_random_over
        )

        if save_status:
            print(
                "\nRandom Oversampling training "
                "data saved successfully."
            )

    # Handle missing training dataset
    except FileNotFoundError:
        print(
            "Error: Required training dataset "
            "file was not found."
        )

    # Handle invalid Random Oversampling data
    except ValueError as e:
        print(
            "Data error during Random Oversampling:",
            e
        )

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during "
            "Random Oversampling:",
            e
        )


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()