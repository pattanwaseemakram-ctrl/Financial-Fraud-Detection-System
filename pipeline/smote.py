import pandas as pd
from imblearn.over_sampling import SMOTE

try:
    # Load training data
    X_train = pd.read_csv("Dataset/X_train_selected.csv")
    y_train = pd.read_csv("Dataset/y_train.csv").squeeze()

    print("Training features shape:", X_train.shape)
    print("Training target shape:", y_train.shape)

    print("\nClass distribution BEFORE SMOTE:")
    print(y_train.value_counts())

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nClass distribution AFTER SMOTE:")
    print(y_train_smote.value_counts())

    print("\nShape BEFORE SMOTE:")
    print(X_train.shape)

    print("\nShape AFTER SMOTE:")
    print(X_train_smote.shape)

    # Save SMOTE-balanced training data
    X_train_smote.to_csv(
        "Dataset/X_train_smote.csv",
        index=False
    )

    y_train_smote.to_csv(
        "Dataset/y_train_smote.csv",
        index=False
    )

    print("\nSMOTE training data saved successfully.")

except FileNotFoundError:
    print("Error: Required training dataset file was not found.")

except ValueError as e:
    print("Data error during SMOTE:", e)

except Exception as e:
    print("Unexpected error during SMOTE:", e)