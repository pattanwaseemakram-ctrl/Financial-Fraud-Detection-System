import os
import pickle
import pandas as pd

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
X_TRAIN_FILE = "Dataset/random oversampling/X_train_random_over.csv"
Y_TRAIN_FILE = "Dataset/random oversampling/y_train_random_over.csv"

X_TEST_FILE = "Dataset/split/X_test_selected.csv"
Y_TEST_FILE = "Dataset/split/y_test.csv"

MODEL_DIR = "balanced_training/models/random_oversampling"


def load_data():
    """Load Random Oversampling training data and original test data."""

    try:
        X_train = pd.read_csv(X_TRAIN_FILE)
        y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()

        X_test = pd.read_csv(X_TEST_FILE)
        y_test = pd.read_csv(Y_TEST_FILE).squeeze()

        print("Random Oversampling training features shape:", X_train.shape)
        print("Random Oversampling training target shape:", y_train.shape)
        print("Test features shape:", X_test.shape)
        print("Test target shape:", y_test.shape)

        print("\nRandom Oversampling training class distribution:")
        print(y_train.value_counts().sort_index())

        print("\nOriginal test class distribution:")
        print(y_test.value_counts().sort_index())

        return X_train, y_train, X_test, y_test

    except FileNotFoundError as error:
        print("File not found:", error)
        raise

    except Exception as error:
        print("Error while loading data:", error)
        raise


def evaluate_model(model, model_name, X_test, y_test):
    """Evaluate a trained model on the untouched test data."""

    try:
        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

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

        # True positives = actual suspicious transactions correctly detected
        true_positives = confusion[1, 1]

        # Total transactions predicted as suspicious
        predicted_suspicious = (predictions == 1).sum()

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(report)

        print("Confusion Matrix:")
        print(confusion)

        print(f"\nROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")

        print(
            f"\nCorrectly detected suspicious transactions: "
            f"{true_positives} out of {sum(y_test == 1)}"
        )

        print(
            f"Total transactions predicted as suspicious: "
            f"{predicted_suspicious}"
        )

        return {
            "Model": model_name,
            "Accuracy": accuracy,
            "Class 1 Precision": report,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "True Positives": true_positives,
            "Predicted Suspicious": predicted_suspicious
        }

    except Exception as error:
        print(f"Error while evaluating {model_name}:", error)
        raise


def save_model(model, model_name):
    """Save trained model as a pickle file."""

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        file_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_random_oversampling.pkl"
        )

        file_path = os.path.join(
            MODEL_DIR,
            file_name
        )

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        print(f"Model saved: {file_path}")

    except Exception as error:
        print(f"Error while saving {model_name}:", error)
        raise


def main():

    try:
        # Load training and test data
        X_train, y_train, X_test, y_test = load_data()

        # Logistic Regression
        logistic_regression = LogisticRegression(
            C=0.01,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )

        logistic_regression.fit(
            X_train,
            y_train
        )

        evaluate_model(
            logistic_regression,
            "Logistic Regression",
            X_test,
            y_test
        )

        save_model(
            logistic_regression,
            "logistic_regression"
        )

        # Decision Tree
        decision_tree = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42
        )

        decision_tree.fit(
            X_train,
            y_train
        )

        evaluate_model(
            decision_tree,
            "Decision Tree",
            X_test,
            y_test
        )

        save_model(
            decision_tree,
            "decision_tree"
        )

        # Random Forest
        random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        random_forest.fit(
            X_train,
            y_train
        )

        evaluate_model(
            random_forest,
            "Random Forest",
            X_test,
            y_test
        )

        save_model(
            random_forest,
            "random_forest"
        )

        # XGBoost
        xgboost_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            random_state=42,
            eval_metric="logloss"
        )

        xgboost_model.fit(
            X_train,
            y_train
        )

        evaluate_model(
            xgboost_model,
            "XGBoost",
            X_test,
            y_test
        )

        save_model(
            xgboost_model,
            "xgboost"
        )

        # SVM
        svm_model = SVC(
            C=10,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        )

        svm_model.fit(
            X_train,
            y_train
        )

        evaluate_model(
            svm_model,
            "SVM",
            X_test,
            y_test
        )

        save_model(
            svm_model,
            "svm"
        )

        # KNN
        knn_model = KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            metric="manhattan"
        )

        knn_model.fit(
            X_train,
            y_train
        )

        evaluate_model(
            knn_model,
            "KNN",
            X_test,
            y_test
        )

        save_model(
            knn_model,
            "knn"
        )

        print("\n" + "=" * 60)
        print("RANDOM OVERSAMPLING TRAINING COMPLETED")
        print("=" * 60)

    except Exception as error:
        print("\nTraining process failed:", error)


if __name__ == "__main__":
    main()