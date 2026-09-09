import os
import pandas as pd
from imblearn.combine import SMOTETomek


# File paths
X_TRAIN_FILE = "Dataset/split/X_train_selected.csv"
Y_TRAIN_FILE = "Dataset/split/y_train.csv"

OUTPUT_DIR = "Dataset/smote tomek"

X_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "X_train_smote_tomek.csv"
)

Y_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "y_train_smote_tomek.csv"
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


def apply_smote_tomek(X_train, y_train):
    """
    Apply SMOTE followed by Tomek Links.
    """

    smote_tomek = SMOTETomek(
        sampling_strategy=0.5,
        random_state=42
    )

    X_resampled, y_resampled = smote_tomek.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled


def save_data(X_resampled, y_resampled):
    """
    Save the SMOTETomek training data.
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

        print(
            "Original training features shape:",
            X_train.shape
        )

        print(
            "Original training target shape:",
            y_train.shape
        )

        # Check class distribution before balancing
        check_distribution(
            y_train,
            "BEFORE SMOTETOMEK:"
        )

        # Apply SMOTETomek
        X_resampled, y_resampled = apply_smote_tomek(
            X_train,
            y_train
        )

        # Check class distribution after balancing
        check_distribution(
            y_resampled,
            "AFTER SMOTETOMEK:"
        )

        print(
            "\nTraining features shape after SMOTETomek:",
            X_resampled.shape
        )

        print(
            "Training target shape after SMOTETomek:",
            y_resampled.shape
        )

        # Save the balanced data
        save_data(
            X_resampled,
            y_resampled
        )

        print("\nSMOTETomek completed successfully.")

        print(
            "Saved features:",
            X_OUTPUT_FILE
        )

        print(
            "Saved target:",
            Y_OUTPUT_FILE
        )

    except Exception as error:

        print(
            "\nError while applying SMOTETomek:",
            error
        )


if __name__ == "__main__":
    main()