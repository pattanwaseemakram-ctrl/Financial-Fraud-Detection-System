import os
import pickle

import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# Input file paths
X_TRAIN_FILE = (
    "Dataset/split/X_train_selected.csv"
)

Y_TRAIN_FILE = (
    "Dataset/split/y_train.csv"
)

X_TEST_FILE = (
    "Dataset/split/X_test_selected.csv"
)

Y_TEST_FILE = (
    "Dataset/split/y_test.csv"
)


# Folder to save SMOTE fine-tuned model
MODEL_DIR = (
    "smote_finetuning/models"
)


# Class containing Logistic Regression fine-tuning methods
class SMOTELogisticFineTuner:

    # Method to load features
    def load_features(self, file_path):
        return pd.read_csv(file_path)

    # Method to load target
    def load_target(self, file_path):
        return pd.read_csv(file_path).squeeze()

    # Method to create pipeline
    def create_pipeline(self):

        smote = SMOTE(
            sampling_strategy=0.5,
            random_state=42
        )

        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        return Pipeline(
            steps=[
                ("smote", smote),
                ("model", model)
            ]
        )

    # Method to create parameter grid
    def create_parameter_grid(self):

        return {
            "model__C": [0.001, 0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
            "model__class_weight": [None, "balanced"]
        }

    # Method to perform GridSearchCV
    def fine_tune(self, pipeline, parameter_grid, X_train, y_train):

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(
            X_train,
            y_train
        )

        return grid_search

    # Method to evaluate model
    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        confusion = confusion_matrix(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities
        )

        print("\nTest Accuracy:")
        print(accuracy)

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        print("\nConfusion Matrix:")
        print(confusion)

        print("\nROC-AUC:")
        print(roc_auc)

        print("\nPR-AUC:")
        print(pr_auc)

        print(
            "\nCorrectly detected suspicious transactions:",
            confusion[1, 1],
            "out of",
            (y_test == 1).sum()
        )

    # Method to save model
    def save_model(self, model):

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        file_path = os.path.join(
            MODEL_DIR,
            "logistic_regression_smote_finetuned.pkl"
        )

        with open(
            file_path,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file
            )

        print(
            "\nModel saved:",
            file_path
        )


# Main function
def main():

    try:

        trainer = SMOTELogisticFineTuner()

        X_train = trainer.load_features(
            X_TRAIN_FILE
        )

        y_train = trainer.load_target(
            Y_TRAIN_FILE
        )

        X_test = trainer.load_features(
            X_TEST_FILE
        )

        y_test = trainer.load_target(
            Y_TEST_FILE
        )

        print(
            "Training features shape:",
            X_train.shape
        )

        print(
            "Training target shape:",
            y_train.shape
        )

        print(
            "Test features shape:",
            X_test.shape
        )

        print(
            "Test target shape:",
            y_test.shape
        )

        print(
            "\nOriginal training class distribution:"
        )

        print(
            y_train.value_counts()
        )

        print(
            "\nTest class distribution:"
        )

        print(
            y_test.value_counts()
        )

        pipeline = trainer.create_pipeline()

        parameter_grid = trainer.create_parameter_grid()

        print(
            "\nStarting Logistic Regression "
            "SMOTE fine-tuning..."
        )

        grid_search = trainer.fine_tune(
            pipeline,
            parameter_grid,
            X_train,
            y_train
        )

        print(
            "\nBest Parameters:"
        )

        print(
            grid_search.best_params_
        )

        print(
            "\nBest Cross-Validation F1:"
        )

        print(
            grid_search.best_score_
        )

        trainer.evaluate_model(
            grid_search.best_estimator_,
            X_test,
            y_test
        )

        trainer.save_model(
            grid_search.best_estimator_
        )

        print(
            "\nLogistic Regression SMOTE "
            "fine-tuning completed successfully."
        )

    except FileNotFoundError:
        print(
            "Error: Required dataset file "
            "was not found."
        )

    except ValueError as e:
        print(
            "Data error during fine-tuning:",
            e
        )

    except Exception as e:
        print(
            "Unexpected error during "
            "fine-tuning:",
            e
        )


if __name__ == "__main__":
    main()