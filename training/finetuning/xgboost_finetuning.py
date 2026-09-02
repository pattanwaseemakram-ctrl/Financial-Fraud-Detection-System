import pandas as pd
import pickle

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

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


# Output file
MODEL_OUTPUT_FILE = (
    "Dataset/xgboost_finetuned.pkl"
)


# Main function
def main():

    try:

        # Load training features
        X_train = pd.read_csv(
            X_TRAIN_FILE
        )

        # Load training target
        y_train = pd.read_csv(
            Y_TRAIN_FILE
        ).squeeze()

        # Load testing features
        X_test = pd.read_csv(
            X_TEST_FILE
        )

        # Load testing target
        y_test = pd.read_csv(
            Y_TEST_FILE
        ).squeeze()


        # Display dataset shapes
        print("Training Features Shape:")
        print(X_train.shape)

        print("\nTraining Target Shape:")
        print(y_train.shape)

        print("\nTesting Features Shape:")
        print(X_test.shape)

        print("\nTesting Target Shape:")
        print(y_test.shape)


        # Display class distributions
        print("\nTraining Class Distribution:")
        print(y_train.value_counts())

        print("\nTesting Class Distribution:")
        print(y_test.value_counts())


        # Create XGBoost model
        model = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )

        print(
            "\nXGBoost model "
            "created successfully."
        )


        # Define hyperparameter grid
        parameter_grid = {

            "n_estimators": [
                100,
                200,
                300
            ],

            "max_depth": [
                3,
                5,
                7
            ],

            "learning_rate": [
                0.01,
                0.1,
                0.2
            ],

            "subsample": [
                0.8,
                1.0
            ],

            "colsample_bytree": [
                0.8,
                1.0
            ],

            "scale_pos_weight": [
                1,
                5,
                10
            ]
        }


        print("\nHyperparameter Grid:")
        print(parameter_grid)


        # Create GridSearchCV
        grid_search = GridSearchCV(

            estimator=model,

            param_grid=parameter_grid,

            scoring="f1",

            cv=5,

            n_jobs=-1,

            verbose=1
        )


        print(
            "\nStarting XGBoost "
            "fine-tuning..."
        )


        # Perform fine-tuning
        grid_search.fit(
            X_train,
            y_train
        )


        print(
            "\nXGBoost fine-tuning "
            "completed successfully."
        )


        # Display best parameters
        print("\nBest Parameters:")

        print(
            grid_search.best_params_
        )


        # Display best CV F1-score
        print(
            "\nBest Cross-Validation F1-score:"
        )

        print(
            grid_search.best_score_
        )


        # Get best model
        best_model = (
            grid_search.best_estimator_
        )


        print(
            "\nBest XGBoost model "
            "selected successfully."
        )


        # Save best model as Pickle
        with open(
            MODEL_OUTPUT_FILE,
            "wb"
        ) as file:

            pickle.dump(
                best_model,
                file
            )


        print(
            "\nFine-tuned model "
            "saved successfully."
        )

        print(
            "Model File:"
        )

        print(
            MODEL_OUTPUT_FILE
        )


        # Make predictions
        y_pred = best_model.predict(
            X_test
        )


        print(
            "\nPredictions completed."
        )


        # Display actual values
        print("\nActual Values:")

        print(
            y_test.values
        )


        # Display predicted values
        print("\nPredicted Values:")

        print(
            y_pred
        )


        # Calculate accuracy
        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        print("\nAccuracy:")

        print(
            accuracy
        )


        # Calculate confusion matrix
        cm = confusion_matrix(
            y_test,
            y_pred
        )


        print("\nConfusion Matrix:")

        print(
            cm
        )


        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )


        print(
            "\nClassification Report:"
        )

        print(
            report
        )


        # Get Class 1 probabilities
        y_probability = (
            best_model.predict_proba(
                X_test
            )[:, 1]
        )


        print(
            "\nFirst 10 Class 1 Probabilities:"
        )

        print(
            y_probability[:10]
        )


        # Calculate ROC-AUC
        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )


        print("\nROC-AUC:")

        print(
            roc_auc
        )


        # Calculate PR-AUC
        pr_auc = average_precision_score(
            y_test,
            y_probability
        )


        print("\nPR-AUC:")

        print(
            pr_auc
        )


        # Final summary
        print("\n========================================")
        print("Fine-Tuned XGBoost Summary")
        print("========================================")

        print("\nBest Parameters:")

        print(
            grid_search.best_params_
        )

        print("\nBest CV F1-score:")

        print(
            grid_search.best_score_
        )

        print("\nTest Accuracy:")

        print(
            accuracy
        )

        print("\nTest ROC-AUC:")

        print(
            roc_auc
        )

        print("\nTest PR-AUC:")

        print(
            pr_auc
        )

        print("\nSaved Model:")

        print(
            MODEL_OUTPUT_FILE
        )


    # Error handling
    except FileNotFoundError:

        print(
            "Error: Required dataset file "
            "was not found.")


    except ValueError as e:

        print(
            "Data error during fine-tuning:",e)


    except Exception as e:

        print(
            "Unexpected error:",e)


# Run main function
if __name__ == "__main__":
    main()