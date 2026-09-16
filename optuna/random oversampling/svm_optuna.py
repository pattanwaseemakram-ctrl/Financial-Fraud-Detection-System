# Import required libraries
import os
import joblib
import optuna
import pandas as pd

from datetime import datetime

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# Project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


# Input dataset paths
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


# Optuna output directory
OPTUNA_DIR = os.path.join(
    PROJECT_ROOT,
    "optuna",
    "random oversampling"
)


# Model output directory
MODEL_DIR = os.path.join(
    OPTUNA_DIR,
    "models"
)


# Saved model path
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "svm_optuna.pkl"
)


# Optuna database path
DATABASE_PATH = os.path.join(
    OPTUNA_DIR,
    "svm_linear_optuna.db"
)


# Optuna settings
N_TRIALS = 15
RANDOM_STATE = 42


def load_data():
    """
    Load training and testing datasets.
    """

    X_train = pd.read_csv(
        X_TRAIN_PATH
    )

    y_train = pd.read_csv(
        Y_TRAIN_PATH
    ).squeeze()

    X_test = pd.read_csv(
        X_TEST_PATH
    )

    y_test = pd.read_csv(
        Y_TEST_PATH
    ).squeeze()

    print("\nData loaded successfully.")

    print(
        f"Training features shape: {X_train.shape}"
    )

    print(
        f"Training target shape: {y_train.shape}"
    )

    print(
        f"Testing features shape: {X_test.shape}"
    )

    print(
        f"Testing target shape: {y_test.shape}"
    )

    print(
        "\nOriginal training class distribution:"
    )

    print(
        y_train.value_counts().sort_index()
    )

    print(
        "\nTesting class distribution:"
    )

    print(
        y_test.value_counts().sort_index()
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


def create_svm_pipeline(
    C,
    class_weight
):
    """
    Create Random Oversampling + Linear SVM pipeline.
    """

    model = Pipeline(
        steps=[
            (
                "random_oversampling",
                RandomOverSampler(
                    sampling_strategy=0.5,
                    random_state=RANDOM_STATE
                )
            ),
            (
                "svm",
                SVC(
                    C=C,
                    kernel="linear",
                    class_weight=class_weight,
                    probability=False,
                    cache_size=1024,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    return model


def objective(
    trial,
    X_train,
    y_train
):
    """
    Optuna objective function.

    Optuna searches for the best C and
    class_weight values using 5-fold
    stratified cross-validation.
    """

    # Search regularization parameter
    C = trial.suggest_float(
        "C",
        0.001,
        100.0,
        log=True
    )

    # Search class weight
    class_weight = trial.suggest_categorical(
        "class_weight",
        [
            None,
            "balanced"
        ]
    )

    # Create Linear SVM pipeline
    model = create_svm_pipeline(
        C=C,
        class_weight=class_weight
    )

    # Create 5-fold stratified CV
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Calculate F1-score
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    # Return mean F1-score
    return scores.mean()


def train_final_model(
    best_params,
    X_train,
    y_train
):
    """
    Train the final Linear SVM model
    using the best Optuna parameters.
    """

    model = create_svm_pipeline(
        C=best_params["C"],
        class_weight=best_params["class_weight"]
    )

    print(
        "\nTraining final Linear SVM model..."
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate the final SVM model.
    """

    # Generate predictions
    y_pred = model.predict(
        X_test
    )

    # Generate decision scores
    y_score = model.decision_function(
        X_test
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Calculate precision
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Calculate recall
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Calculate F1-score
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(
        y_test,
        y_score
    )

    # Calculate PR-AUC
    pr_auc = average_precision_score(
        y_test,
        y_score
    )

    # Calculate confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    # Extract confusion matrix values
    true_negative = matrix[0, 0]
    false_positive = matrix[0, 1]
    false_negative = matrix[1, 0]
    true_positive = matrix[1, 1]

    # Count suspicious transactions
    total_suspicious = int(
        (y_test == 1).sum()
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "LINEAR SVM + RANDOM OVERSAMPLING"
    )

    print(
        "OPTUNA FINAL TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall   : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-Score : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        matrix
    )

    print(
        "\nConfusion Matrix Details:"
    )

    print(
        f"True Negatives : {true_negative}"
    )

    print(
        f"False Positives: {false_positive}"
    )

    print(
        f"False Negatives: {false_negative}"
    )

    print(
        f"True Positives : {true_positive}"
    )

    print(
        "\nSuspicious Activity Detection:"
    )

    print(
        f"Suspicious transactions detected: "
        f"{true_positive}/{total_suspicious}"
    )

    print(
        f"False suspicious predictions: "
        f"{false_positive}"
    )


def main():
    """
    Main function.
    """

    try:

        print(
            "=" * 60
        )

        print(
            "SVM"
        )

        print(
            "RANDOM OVERSAMPLING + OPTUNA"
        )

        print(
            "=" * 60
        )

        # Create directories
        os.makedirs(
            OPTUNA_DIR,
            exist_ok=True
        )

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        # Load datasets
        (
            X_train,
            y_train,
            X_test,
            y_test
        ) = load_data()

        print(
            "\nStarting Optuna optimization..."
        )

        print(
            f"Number of trials: {N_TRIALS}"
        )

        print(
            "Cross-validation: 5-fold Stratified CV"
        )

        print(
            "Optimization metric: Class 1 F1-score"
        )

        print(
            "Random Oversampling strategy: 0.5"
        )

        print(
            "SVM kernel: linear"
        )

        print(
            "Probability calculation: Disabled"
        )

        # Create unique study name
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        study_name = (
            "svm_linear_random_oversampling_"
            + timestamp
        )

        print(
            f"\nOptuna study name: {study_name}"
        )

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=f"sqlite:///{DATABASE_PATH}",
            load_if_exists=False
        )

        # Run Optuna
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train
            ),
            n_trials=N_TRIALS,
            show_progress_bar=True
        )

        # Display best result
        print(
            "\n" + "=" * 60
        )

        print(
            "OPTUNA OPTIMIZATION COMPLETED"
        )

        print(
            "=" * 60
        )

        print(
            f"Best Cross-Validation F1-score: "
            f"{study.best_value:.4f}"
        )

        print(
            "\nBest Hyperparameters:"
        )

        for parameter, value in (
            study.best_params.items()
        ):

            print(
                f"{parameter}: {value}"
            )

        # Train final model
        final_model = train_final_model(
            study.best_params,
            X_train,
            y_train
        )

        # Save final model
        joblib.dump(
            final_model,
            MODEL_PATH
        )

        print(
            "\nFinal model saved successfully:"
        )

        print(
            MODEL_PATH
        )

        # Evaluate final model
        evaluate_model(
            final_model,
            X_test,
            y_test
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "SVM OPTUNA EXPERIMENT "
            "COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 60
        )

    except FileNotFoundError as e:

        print(
            "\nFile not found:"
        )

        print(e)

    except ValueError as e:

        print(
            "\nData or parameter error:"
        )

        print(e)

    except Exception as e:

        print(
            "\nUnexpected error:"
        )

        print(e)


# Run the main function
if __name__ == "__main__":
    main()