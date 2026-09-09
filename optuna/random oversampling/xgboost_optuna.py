# Import required libraries
import os
import joblib
import optuna
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from xgboost import XGBClassifier

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

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_optuna_v2.pkl"
)


# Optuna database path
DATABASE_PATH = os.path.join(
    OPTUNA_DIR,
    "xgboost_optuna_v2.db"
)


# Optuna settings
N_TRIALS = 30
RANDOM_STATE = 42


def load_data():
    """
    Load training and testing datasets.
    """

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    print("\nData loaded successfully.")

    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Testing target shape: {y_test.shape}")

    print("\nOriginal training class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, y_train, X_test, y_test


def objective(trial, X_train, y_train):
    """
    Optuna objective function.

    Optuna searches for the best XGBoost
    hyperparameters using 5-fold cross-validation.
    """

    # Hyperparameter search space
    n_estimators = trial.suggest_int(
        "n_estimators",
        100,
        500,
        step=50
    )

    max_depth = trial.suggest_int(
        "max_depth",
        3,
        10
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        0.01,
        0.3,
        log=True
    )

    subsample = trial.suggest_float(
        "subsample",
        0.6,
        1.0
    )

    colsample_bytree = trial.suggest_float(
        "colsample_bytree",
        0.6,
        1.0
    )

    min_child_weight = trial.suggest_int(
        "min_child_weight",
        1,
        10
    )

    gamma = trial.suggest_float(
        "gamma",
        0.0,
        5.0
    )

    reg_alpha = trial.suggest_float(
        "reg_alpha",
        0.0,
        1.0
    )

    reg_lambda = trial.suggest_float(
        "reg_lambda",
        0.1,
        10.0,
        log=True
    )

    scale_pos_weight = trial.suggest_float(
        "scale_pos_weight",
        1.0,
        10.0
    )

    # Random Oversampling + XGBoost
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
                "xgboost",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    gamma=gamma,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1
                )
            )
        ]
    )

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Optimize Class 1 F1-score
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    return scores.mean()


def train_final_model(best_params, X_train, y_train):
    """
    Train the final XGBoost model
    using the best Optuna parameters.
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
                "xgboost",
                XGBClassifier(
                    n_estimators=best_params["n_estimators"],
                    max_depth=best_params["max_depth"],
                    learning_rate=best_params["learning_rate"],
                    subsample=best_params["subsample"],
                    colsample_bytree=best_params["colsample_bytree"],
                    min_child_weight=best_params["min_child_weight"],
                    gamma=best_params["gamma"],
                    reg_alpha=best_params["reg_alpha"],
                    reg_lambda=best_params["reg_lambda"],
                    scale_pos_weight=best_params["scale_pos_weight"],
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the optimized model on the untouched test data.
    """

    # Generate predictions
    y_pred = model.predict(X_test)

    # Generate probability scores
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
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

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    # Display results
    print("\n" + "=" * 60)
    print("OPTUNA V2 + RANDOM OVERSAMPLING")
    print("XGBOOST TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    # Extract TP and FP
    true_positive = matrix[1, 1]
    false_positive = matrix[0, 1]

    total_suspicious = sum(
        y_test == 1
    )

    print("\nSuspicious Activity Detection:")
    print(f"True Positives : {true_positive}")
    print(f"False Positives: {false_positive}")

    print(
        f"Suspicious transactions detected: "
        f"{true_positive}/{total_suspicious}"
    )


def main():
    """
    Main function.
    """

    try:

        print("=" * 60)
        print("XGBOOST")
        print("RANDOM OVERSAMPLING + OPTUNA V2")
        print("=" * 60)

        # Create Optuna output directory
        os.makedirs(
            OPTUNA_DIR,
            exist_ok=True
        )

        # Create model output directory
        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        # Load data
        X_train, y_train, X_test, y_test = load_data()

        print("\nStarting Optuna optimization...")
        print(f"Number of trials: {N_TRIALS}")
        print("Cross-validation: 5-fold Stratified CV")
        print("Optimization metric: Class 1 F1-score")
        print(
            "Hyperparameters: "
            "n_estimators, max_depth, learning_rate, "
            "subsample, colsample_bytree, min_child_weight, "
            "gamma, reg_alpha, reg_lambda, scale_pos_weight"
        )

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name="xgboost_random_oversampling_v2",
            storage=f"sqlite:///{DATABASE_PATH}",
            load_if_exists=True
        )

        # Run Optuna
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train
            ),
            n_trials=N_TRIALS
        )

        # Display best results
        print("\n" + "=" * 60)
        print("OPTUNA V2 OPTIMIZATION COMPLETED")
        print("=" * 60)

        print(
            f"Best Cross-Validation F1-score: "
            f"{study.best_value:.4f}"
        )

        print("\nBest Hyperparameters:")

        for parameter, value in study.best_params.items():
            print(
                f"{parameter}: {value}"
            )

        # Train final model
        print(
            "\nTraining final model using "
            "best Optuna parameters..."
        )

        final_model = train_final_model(
            study.best_params,
            X_train,
            y_train
        )

        # Save model
        joblib.dump(
            final_model,
            MODEL_PATH
        )

        print(
            f"\nModel saved successfully:\n"
            f"{MODEL_PATH}"
        )

        # Evaluate on untouched test data
        evaluate_model(
            final_model,
            X_test,
            y_test
        )

        print(
            "\nXGBoost Optuna V2 "
            "optimization completed successfully."
        )

    except ValueError as e:

        print(
            "\nData or parameter error:",
            e
        )

    except Exception as e:

        print(
            "\nUnexpected error:",
            e
        )


# Run the main function
if __name__ == "__main__":
    main()