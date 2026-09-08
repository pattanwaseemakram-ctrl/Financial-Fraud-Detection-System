import os
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE


# File paths
X_TRAIN_FILE = "Dataset/split/X_train_selected.csv"
Y_TRAIN_FILE = "Dataset/split/y_train.csv"

OUTPUT_DIR = "Dataset/borderline smote"

X_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "X_train_borderline_smote.csv"
)

Y_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "y_train_borderline_smote.csv"
)


def load_data():
    """
    Load the selected training features and training target.
    """

    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()

    return X_train, y_train


def check_distribution(y_train, message):
    """
    Display the class distribution.
    """

    print(f"\n{message}")
    print(y_train.value_counts().sort_index())


def apply_borderline_smote(X_train, y_train):
    """
    Apply Borderline-SMOTE to the training data.
    """

    borderline_smote = BorderlineSMOTE(
        sampling_strategy=0.5,
        random_state=42
    )

    X_resampled, y_resampled = borderline_smote.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled


def save_data(X_resampled, y_resampled):
    """
    Save the Borderline-SMOTE training data.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    X_resampled.to_csv(
        X_OUTPUT_FILE,
        index=False
    )

    pd.Series(y_resampled).to_csv(
        Y_OUTPUT_FILE,
        index=False,
        header=["Suspicious Activity Flag"]
    )


def main():

    try:

        # Load training data
        X_train, y_train = load_data()

        print("Original training features shape:", X_train.shape)
        print("Original training target shape:", y_train.shape)

        # Check class distribution before balancing
        check_distribution(
            y_train,
            "BEFORE BORDERLINE-SMOTE:"
        )

        # Apply Borderline-SMOTE
        X_resampled, y_resampled = apply_borderline_smote(
            X_train,
            y_train
        )

        # Check class distribution after balancing
        check_distribution(
            y_resampled,
            "AFTER BORDERLINE-SMOTE:"
        )

        print(
            "\nTraining features shape after Borderline-SMOTE:",
            X_resampled.shape
        )

        print(
            "Training target shape after Borderline-SMOTE:",
            y_resampled.shape
        )

        # Save the balanced data
        save_data(
            X_resampled,
            y_resampled
        )

        print("\nBorderline-SMOTE completed successfully.")
        print("Saved features:", X_OUTPUT_FILE)
        print("Saved target:", Y_OUTPUT_FILE)

    except Exception as error:

        print(
            "\nError while applying Borderline-SMOTE:",
            error
        )


if __name__ == "__main__":
    main()