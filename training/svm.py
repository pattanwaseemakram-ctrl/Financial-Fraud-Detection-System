import pandas as pd

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# Input files
X_TRAIN_FILE = "Dataset/X_train_selected.csv"
Y_TRAIN_FILE = "Dataset/y_train.csv"

X_TEST_FILE = "Dataset/X_test_selected.csv"
Y_TEST_FILE = "Dataset/y_test.csv"


# Main function to control the SVM workflow
def main():

    try:

        # Load training features
        X_train = pd.read_csv(X_TRAIN_FILE)

        # Load training target
        y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()

        # Load testing features
        X_test = pd.read_csv(X_TEST_FILE)

        # Load testing target
        y_test = pd.read_csv(Y_TEST_FILE).squeeze()


        # Display dataset shapes
        print("Training Features Shape:")
        print(X_train.shape)

        print("\nTraining Target Shape:")
        print(y_train.shape)

        print("\nTesting Features Shape:")
        print(X_test.shape)

        print("\nTesting Target Shape:")
        print(y_test.shape)


        # Display training class distribution
        print("\nTraining Class Distribution:")
        print(y_train.value_counts())

        # Display testing class distribution
        print("\nTesting Class Distribution:")
        print(y_test.value_counts())


        # Create SVM model
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        )

        print(
            "\nSVM model "
            "created successfully.")


        # Train SVM model
        model.fit(
            X_train,
            y_train)

        print(
            "\nSVM training "
            "completed successfully.")


        # Make predictions
        y_pred = model.predict(
            X_test)

        print("\nPredictions completed.")


        # Display actual values
        print("\nActual Values:")
        print(y_test.values)


        # Display predicted values
        print("\nPredicted Values:")
        print(y_pred)


        # Calculate accuracy
        accuracy = accuracy_score(
            y_test,
            y_pred)

        print("\nAccuracy:")
        print(accuracy)


        # Calculate confusion matrix
        cm = confusion_matrix(
            y_test,
            y_pred)

        print("\nConfusion Matrix:")
        print(cm)


        # Generate classification report
        report = classification_report(
            y_test,
            y_pred)

        print("\nClassification Report:")
        print(report)


        # Get probability predictions
        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        print("\nFirst 10 Class 1 Probabilities:")
        print(y_probability[:10])


        # Calculate ROC-AUC
        roc_auc = roc_auc_score(
            y_test,
            y_probability)

        print("\nROC-AUC:")
        print(roc_auc)


        # Calculate PR-AUC
        pr_auc = average_precision_score(
            y_test,
            y_probability)

        print("\nPR-AUC:")
        print(pr_auc)


    except FileNotFoundError:
        print(
            "Error: Required dataset file "
            "was not found.")

    except ValueError as e:
        print(
            "Data error during "
            "SVM training:",e)

    except Exception as e:
        print("Unexpected error:",e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()