import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Dataset paths
X_TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "Dataset",
    "split",
    "X_train_selected.csv"
)

Y_TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "Dataset",
    "split",
    "y_train.csv"
)

X_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "Dataset",
    "split",
    "X_test_selected.csv"
)

Y_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "Dataset",
    "split",
    "y_test.csv"
)


# Model output path
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "balanced_finetuning",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_oversampling_logistic_finetuned.pkl"
)


def load_data():
    """Load training and testing data."""

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return X_train, y_train, X_test, y_test


def create_pipeline():
    """Create a pipeline with Random Oversampling and Logistic Regression."""

    pipeline = Pipeline(
        steps=[
            (
                "oversampling",
                RandomOverSampler(
                    sampling_strategy=0.5,
                    random_state=42
                )
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    return pipeline


def fine_tune_model(X_train, y_train):
    """Fine-tune Logistic Regression using GridSearchCV."""

    pipeline = create_pipeline()

    parameter_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__solver": ["lbfgs", "liblinear"],
        "model__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    return grid_search


def evaluate_model(model, X_test, y_test):
    """Evaluate the fine-tuned model on untouched test data."""

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nFine-Tuned Logistic Regression Results")
    print("----------------------------------------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nTrue Positives  : {cm[1, 1]}")
    print(f"False Positives : {cm[0, 1]}")


def main():

    try:

        print("Loading data...")

        X_train, y_train, X_test, y_test = load_data()

        print(f"Training data shape : {X_train.shape}")
        print(f"Testing data shape  : {X_test.shape}")

        print("\nOriginal training class distribution:")
        print(y_train.value_counts())

        print("\nStarting fine-tuning...")
        print("Balancing technique : Random Oversampling")
        print("Sampling strategy   : 0.5")
        print("Cross-validation    : 5-fold Stratified CV")
        print("Optimization metric : Class 1 F1-score")

        grid_search = fine_tune_model(
            X_train,
            y_train
        )

        print("\nFine-tuning completed.")

        print("\nBest CV F1-score:")
        print(
            f"{grid_search.best_score_:.4f}"
        )

        print("\nBest Parameters:")

        for parameter, value in grid_search.best_params_.items():
            print(
                f"{parameter}: {value}"
            )

        print("\nTraining final model...")

        final_model = grid_search.best_estimator_

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        joblib.dump(
            final_model,
            MODEL_PATH
        )

        print("\nModel saved to:")
        print(MODEL_PATH)

        evaluate_model(
            final_model,
            X_test,
            y_test
        )

    except Exception as e:

        print("\nError occurred:")
        print(e)


if __name__ == "__main__":
    main()