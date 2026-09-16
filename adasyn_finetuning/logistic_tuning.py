import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline


# File paths
X_TRAIN_FILE = "Dataset/split/X_train_selected.csv"
Y_TRAIN_FILE = "Dataset/split/y_train.csv"

X_TEST_FILE = "Dataset/split/X_test_selected.csv"
Y_TEST_FILE = "Dataset/split/y_test.csv"

MODEL_DIR = "adasyn_finetuning/models"


def main():

    try:

        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Load training data
        X_train = pd.read_csv(X_TRAIN_FILE)
        y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()

        # Load testing data
        X_test = pd.read_csv(X_TEST_FILE)
        y_test = pd.read_csv(Y_TEST_FILE).squeeze()

        print("Training data shape:", X_train.shape)
        print("Training target shape:", y_train.shape)

        print("Testing data shape:", X_test.shape)
        print("Testing target shape:", y_test.shape)

        print("\nOriginal training class distribution:")
        print(y_train.value_counts())

        # ADASYN + Logistic Regression pipeline
        pipeline = Pipeline([
            (
                "adasyn",
                ADASYN(
                    sampling_strategy=0.5,
                    random_state=42
                )
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ])

        # Hyperparameter grid
        parameter_grid = {
            "model__C": [0.001, 0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
            "model__class_weight": [None, "balanced"]
        }

        print("\nStarting ADASYN + Logistic Regression fine-tuning...")

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        # Train model
        grid_search.fit(X_train, y_train)

        print("\nBest Parameters:")
        print(grid_search.best_params_)

        print("\nBest Cross-Validation F1 Score:")
        print(grid_search.best_score_)

        # Get best model
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)

        # Probability scores
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        print("\nTest Set Results")
        print("=" * 50)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        # Save model
        model_path = os.path.join(
            MODEL_DIR,
            "logistic_regression_adasyn_finetuned.pkl"
        )

        joblib.dump(best_model, model_path)

        print("\nModel saved to:")
        print(model_path)

        print("\nADASYN + Logistic Regression fine-tuning completed successfully.")

    except Exception as e:

        print("\nError occurred:")
        print(e)


if __name__ == "__main__":
    main()