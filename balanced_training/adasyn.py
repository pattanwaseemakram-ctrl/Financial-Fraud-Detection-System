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
X_TRAIN_FILE = "Dataset/adasyn/X_train_adasyn.csv"
Y_TRAIN_FILE = "Dataset/adasyn/y_train_adasyn.csv"

X_TEST_FILE = "Dataset/split/X_test_selected.csv"
Y_TEST_FILE = "Dataset/split/y_test.csv"

MODEL_DIR = "balanced_training/models/adasyn"


def load_data():

    X_train = pd.read_csv(X_TRAIN_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()

    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).squeeze()

    print("ADASYN training features shape:", X_train.shape)
    print("ADASYN training target shape:", y_train.shape)
    print("Test features shape:", X_test.shape)
    print("Test target shape:", y_test.shape)

    print("\nADASYN training class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nOriginal test class distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, y_train, X_test, y_test


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    # Get prediction probabilities for ROC-AUC and PR-AUC
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        digits=2
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

    # True positives
    true_positives = confusion[1, 1]

    # Total transactions predicted as suspicious
    predicted_suspicious = np.sum(predictions == 1)

    print("\nAccuracy:")
    print(round(accuracy, 4))

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(confusion)

    print("\nROC-AUC:")
    print(round(roc_auc, 4))

    print("\nPR-AUC:")
    print(round(pr_auc, 4))

    print(
        "\nCorrectly detected suspicious transactions:",
        true_positives,
        "out of",
        201
    )

    print(
        "Total transactions predicted as suspicious:",
        predicted_suspicious
    )


def save_model(model, filename):

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

    print("Model saved:", model_path)


def train_logistic_regression(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION - ADASYN")
    print("=" * 60)

    model = LogisticRegression(
        C=0.01,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(
        model,
        "logistic_regression_adasyn.pkl"
    )


def train_decision_tree(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("DECISION TREE - ADASYN")
    print("=" * 60)

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(
        model,
        "decision_tree_adasyn.pkl"
    )


def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("RANDOM FOREST - ADASYN")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(
        model,
        "random_forest_adasyn.pkl"
    )


def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("XGBOOST - ADASYN")
    print("=" * 60)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(
        model,
        "xgboost_adasyn.pkl"
    )


def train_svm(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("SVM - ADASYN")
    print("=" * 60)

    model = SVC(
        C=10,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(
        model,
        "svm_adasyn.pkl"
    )


def train_knn(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("KNN - ADASYN")
    print("=" * 60)

    model = KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
        metric="manhattan"
    )

    model.fit(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(
        model,
        "knn_adasyn.pkl"
    )


def main():

    try:

        # Load ADASYN training data and original test data
        X_train, y_train, X_test, y_test = load_data()

        # Train all six models
        train_logistic_regression(
            X_train,
            y_train,
            X_test,
            y_test
        )

        train_decision_tree(
            X_train,
            y_train,
            X_test,
            y_test
        )

        train_random_forest(
            X_train,
            y_train,
            X_test,
            y_test
        )

        train_xgboost(
            X_train,
            y_train,
            X_test,
            y_test
        )

        train_svm(
            X_train,
            y_train,
            X_test,
            y_test
        )

        train_knn(
            X_train,
            y_train,
            X_test,
            y_test
        )

        print("\n" + "=" * 60)
        print("ADASYN MODEL TRAINING AND EVALUATION COMPLETED")
        print("=" * 60)

    except FileNotFoundError as error:

        print("\nFile not found:")
        print(error)

    except Exception as error:

        print("\nAn error occurred:")
        print(error)


if __name__ == "__main__":
    main()