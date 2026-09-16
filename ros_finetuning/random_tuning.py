# Random Oversampling + Random Forest Fine-Tuning

import os
import joblib
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# File paths
X_TRAIN_PATH = "Dataset/split/X_train_selected.csv"
Y_TRAIN_PATH = "Dataset/split/y_train.csv"

X_TEST_PATH = "Dataset/split/X_test_selected.csv"
Y_TEST_PATH = "Dataset/split/y_test.csv"

MODEL_DIR = "balanced_finetuning/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_oversampling_random_forest_finetuned.pkl"
)


def main():

    try:
        # Create model directory if it does not exist
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Load training and testing data
        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()

        print("Training data shape:", X_train.shape)
        print("Testing data shape:", X_test.shape)

        print("\nOriginal training class distribution:")
        print(y_train.value_counts())

        # Create Random Oversampling
        oversampler = RandomOverSampler(
            sampling_strategy=0.5,
            random_state=42
        )

        # Create Random Forest model
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

        # Create pipeline
        # Oversampling happens inside each CV fold
        pipeline = Pipeline([
            ("oversampler", oversampler),
            ("model", model)
        ])

        # Define hyperparameter grid
        parameter_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": [None, "balanced"]
        }

        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        print("\nStarting Random Forest balanced fine-tuning...")

        # Perform hyperparameter tuning
        grid_search.fit(X_train, y_train)

        # Display best parameters
        print("\nBest Parameters:")
        print(grid_search.best_params_)

        print("\nBest Cross-Validation F1 Score:")
        print(f"{grid_search.best_score_:.4f}")

        # Get best model
        best_model = grid_search.best_estimator_

        # Make predictions on untouched test data
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

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
            y_prob
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        # Display results
        print("\nTest Set Results")
        print("-------------------------")
        print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
        print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
        print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        # Extract confusion matrix values
        tn, fp, fn, tp = cm.ravel()

        print("\nConfusion Matrix Details:")
        print(f"True Negatives  (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives  (TP): {tp}")

        # Save the best model
        joblib.dump(
            best_model,
            MODEL_PATH
        )

        print("\nBest model saved successfully:")
        print(MODEL_PATH)

    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()