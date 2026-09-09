import os
import pickle

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# Input file paths
X_TRAIN_FILE = (
    "Dataset/smote/X_train_smote.csv"
)

Y_TRAIN_FILE = (
    "Dataset/smote/y_train_smote.csv"
)

X_TEST_FILE = (
    "Dataset/split/X_test_selected.csv"
)

Y_TEST_FILE = (
    "Dataset/split/y_test.csv"
)


# Folder to save SMOTE trained models
MODEL_DIR = (
    "balanced_training/models/smote"
)


# Class containing all SMOTE model training methods
class SMOTEModelTrainer:

    # Method to load features
    def load_features(self, file_path):
        return pd.read_csv(file_path)

    # Method to load target
    def load_target(self, file_path):
        return pd.read_csv(file_path).squeeze()

    # Method to create Logistic Regression model
    def create_logistic_regression(self):
        return LogisticRegression(
            C=0.01,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )

    # Method to create Decision Tree model
    def create_decision_tree(self):
        return DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42
        )

    # Method to create Random Forest model
    def create_random_forest(self):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        )

    # Method to create XGBoost model
    def create_xgboost(self):
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            random_state=42,
            eval_metric="logloss"
        )

    # Method to create SVM model
    def create_svm(self):
        return SVC(
            C=10,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        )

    # Method to create KNN model
    def create_knn(self):
        return KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            metric="manhattan"
        )

    # Method to train model
    def train_model(self, model, X_train, y_train):
        model.fit(
            X_train,
            y_train
        )
        return model

    # Method to save trained model
    def save_model(self, model, file_name):

        # Create model folder if it does not exist
        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        file_path = os.path.join(
            MODEL_DIR,
            file_name
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

    # Method to evaluate model
    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):
        predictions = model.predict(X_test)

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

        # Correctly detected suspicious transactions
        true_positives = confusion[1, 1]

        # Total transactions predicted as suspicious
        predicted_suspicious = np.sum(
            predictions == 1
        )

        print("\nAccuracy:")
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
            "\nCorrectly detected suspicious "
            "transactions:",
            true_positives,
            "out of",
            np.sum(y_test == 1)
        )

        print(
            "Total transactions predicted "
            "as suspicious:",
            predicted_suspicious
        )


# Main function to control the SMOTE training workflow
def main():

    try:

        # Create SMOTEModelTrainer object
        trainer = SMOTEModelTrainer()

        # Load SMOTE training features
        X_train = trainer.load_features(
            X_TRAIN_FILE
        )

        # Load SMOTE training target
        y_train = trainer.load_target(
            Y_TRAIN_FILE
        )

        # Load original test features
        X_test = trainer.load_features(
            X_TEST_FILE
        )

        # Load original test target
        y_test = trainer.load_target(
            Y_TEST_FILE
        )

        # Display training and testing shapes
        print(
            "SMOTE training features shape:",
            X_train.shape
        )

        print(
            "SMOTE training target shape:",
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

        # Display training class distribution
        print(
            "\nSMOTE training class distribution:"
        )

        print(
            y_train.value_counts()
        )

        # Display test class distribution
        print(
            "\nOriginal test class distribution:"
        )

        print(
            y_test.value_counts()
        )


        # Logistic Regression
        print(
            "\n" + "=" * 60
        )

        print(
            "LOGISTIC REGRESSION - SMOTE"
        )

        print(
            "=" * 60
        )

        logistic_model = (
            trainer.create_logistic_regression()
        )

        logistic_model = trainer.train_model(
            logistic_model,
            X_train,
            y_train
        )

        trainer.evaluate_model(
            logistic_model,
            X_test,
            y_test
        )

        trainer.save_model(
            logistic_model,
            "logistic_regression_smote.pkl"
        )


        # Decision Tree
        print(
            "\n" + "=" * 60
        )

        print(
            "DECISION TREE - SMOTE"
        )

        print(
            "=" * 60
        )

        decision_tree_model = (
            trainer.create_decision_tree()
        )

        decision_tree_model = trainer.train_model(
            decision_tree_model,
            X_train,
            y_train
        )

        trainer.evaluate_model(
            decision_tree_model,
            X_test,
            y_test
        )

        trainer.save_model(
            decision_tree_model,
            "decision_tree_smote.pkl"
        )


        # Random Forest
        print(
            "\n" + "=" * 60
        )

        print(
            "RANDOM FOREST - SMOTE"
        )

        print(
            "=" * 60
        )

        random_forest_model = (
            trainer.create_random_forest()
        )

        random_forest_model = trainer.train_model(
            random_forest_model,
            X_train,
            y_train
        )

        trainer.evaluate_model(
            random_forest_model,
            X_test,
            y_test
        )

        trainer.save_model(
            random_forest_model,
            "random_forest_smote.pkl"
        )


        # XGBoost
        print(
            "\n" + "=" * 60
        )

        print(
            "XGBOOST - SMOTE"
        )

        print(
            "=" * 60
        )

        xgboost_model = (
            trainer.create_xgboost()
        )

        xgboost_model = trainer.train_model(
            xgboost_model,
            X_train,
            y_train
        )

        trainer.evaluate_model(
            xgboost_model,
            X_test,
            y_test
        )

        trainer.save_model(
            xgboost_model,
            "xgboost_smote.pkl"
        )


        # SVM
        print(
            "\n" + "=" * 60
        )

        print(
            "SVM - SMOTE"
        )

        print(
            "=" * 60
        )

        svm_model = trainer.create_svm()

        svm_model = trainer.train_model(
            svm_model,
            X_train,
            y_train
        )

        trainer.evaluate_model(
            svm_model,
            X_test,
            y_test
        )

        trainer.save_model(
            svm_model,
            "svm_smote.pkl"
        )


        # KNN
        print(
            "\n" + "=" * 60
        )

        print(
            "KNN - SMOTE"
        )

        print(
            "=" * 60
        )

        knn_model = trainer.create_knn()

        knn_model = trainer.train_model(
            knn_model,
            X_train,
            y_train
        )

        trainer.evaluate_model(
            knn_model,
            X_test,
            y_test
        )

        trainer.save_model(
            knn_model,
            "knn_smote.pkl"
        )


        print(
            "\nSMOTE model training and "
            "evaluation completed successfully."
        )

    # Handle missing dataset files
    except FileNotFoundError:
        print(
            "Error: Required dataset file "
            "was not found."
        )

    # Handle invalid data
    except ValueError as e:
        print(
            "Data error during model training:",
            e
        )

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during "
            "model training:",
            e
        )


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()