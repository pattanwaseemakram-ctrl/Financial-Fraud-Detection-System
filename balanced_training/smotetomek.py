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


# File paths
X_TRAIN_FILE = "Dataset/smote tomek/X_train_smote_tomek.csv"
Y_TRAIN_FILE = "Dataset/smote tomek/y_train_smote_tomek.csv"

X_TEST_FILE = "Dataset/split/X_test_selected.csv"
Y_TEST_FILE = "Dataset/split/y_test.csv"

MODEL_DIR = "balanced_training/models/smote_tomek"


def load_data():
    """
    Load SMOTETomek training data and original test data.
    """

    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()

    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).squeeze()

    return X_train, y_train, X_test, y_test


def save_model(model, filename):
    """
    Save the trained model as a pickle file.
    """

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print("Saved model:", model_path)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model on the original untouched test data.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
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

    confusion = confusion_matrix(
        y_test,
        predictions
    )

    true_positives = confusion[1, 1]

    predicted_suspicious = np.sum(
        predictions == 1
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("Confusion Matrix:")
    print(confusion)

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"\nCorrectly detected suspicious transactions: "
        f"{true_positives} out of {np.sum(y_test == 1)}"
    )

    print(
        f"Total transactions predicted as suspicious: "
        f"{predicted_suspicious}"
    )


def main():

    try:

        # Load data
        X_train, y_train, X_test, y_test = load_data()

        print(
            "SMOTETomek training features shape:",
            X_train.shape
        )

        print(
            "SMOTETomek training target shape:",
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

        print("\nSMOTETomek training class distribution:")
        print(
            y_train.value_counts().sort_index()
        )

        print("\nOriginal test class distribution:")
        print(
            y_test.value_counts().sort_index()
        )


        # Logistic Regression
        logistic_regression = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        logistic_regression.fit(
            X_train,
            y_train
        )

        evaluate_model(
            logistic_regression,
            X_test,
            y_test,
            "SMOTETomek + Logistic Regression"
        )

        save_model(
            logistic_regression,
            "logistic_regression_smote_tomek.pkl"
        )


        # Decision Tree
        decision_tree = DecisionTreeClassifier(
            random_state=42
        )

        decision_tree.fit(
            X_train,
            y_train
        )

        evaluate_model(
            decision_tree,
            X_test,
            y_test,
            "SMOTETomek + Decision Tree"
        )

        save_model(
            decision_tree,
            "decision_tree_smote_tomek.pkl"
        )


        # Random Forest
        random_forest = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        random_forest.fit(
            X_train,
            y_train
        )

        evaluate_model(
            random_forest,
            X_test,
            y_test,
            "SMOTETomek + Random Forest"
        )

        save_model(
            random_forest,
            "random_forest_smote_tomek.pkl"
        )


        # XGBoost
        xgboost_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )

        xgboost_model.fit(
            X_train,
            y_train
        )

        evaluate_model(
            xgboost_model,
            X_test,
            y_test,
            "SMOTETomek + XGBoost"
        )

        save_model(
            xgboost_model,
            "xgboost_smote_tomek.pkl"
        )


        # SVM
        svm_model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        )

        svm_model.fit(
            X_train,
            y_train
        )

        evaluate_model(
            svm_model,
            X_test,
            y_test,
            "SMOTETomek + SVM"
        )

        save_model(
            svm_model,
            "svm_smote_tomek.pkl"
        )


        # KNN
        knn_model = KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski"
        )

        knn_model.fit(
            X_train,
            y_train
        )

        evaluate_model(
            knn_model,
            X_test,
            y_test,
            "SMOTETomek + KNN"
        )

        save_model(
            knn_model,
            "knn_smote_tomek.pkl"
        )


        print("\n" + "=" * 60)
        print("SMOTETOMEK TRAINING COMPLETED")
        print("=" * 60)


    except Exception as error:

        print(
            "\nError during SMOTETomek training:",
            error
        )


if __name__ == "__main__":
    main()