import os
import joblib
import optuna

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import RandomOverSampler


# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


# Output paths
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "optuna",
    "random oversampling",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "knn_optuna_v2.pkl"
)


# Optuna settings
N_TRIALS = 30

STUDY_NAME = "knn_random_oversampling_v2"

DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "optuna",
    "random oversampling",
    "knn_random_oversampling_v2.db"
)


def load_data():
    """Load training and testing data."""

    X_train = __import__("pandas").read_csv(X_TRAIN_PATH)
    y_train = __import__("pandas").read_csv(Y_TRAIN_PATH).squeeze()

    X_test = __import__("pandas").read_csv(X_TEST_PATH)
    y_test = __import__("pandas").read_csv(Y_TEST_PATH).squeeze()

    return X_train, y_train, X_test, y_test


def apply_random_oversampling(X_train, y_train):
    """Apply Random Oversampling only to training data."""

    ros = RandomOverSampler(
        sampling_strategy=0.5,
        random_state=42
    )

    X_train_ros, y_train_ros = ros.fit_resample(
        X_train,
        y_train
    )

    return X_train_ros, y_train_ros


def objective(trial, X_train, y_train):
    """Optuna objective function."""

    n_neighbors = trial.suggest_int(
        "n_neighbors",
        3,
        30
    )

    weights = trial.suggest_categorical(
        "weights",
        ["uniform", "distance"]
    )

    metric = trial.suggest_categorical(
        "metric",
        ["euclidean", "manhattan", "minkowski"]
    )

    p = trial.suggest_int(
        "p",
        1,
        2
    )

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

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
    """Train the final KNN model using the best parameters."""

    model = KNeighborsClassifier(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        metric=best_params["metric"],
        p=best_params["p"]
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the final model on the test data."""

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix
    )

    y_pred = model.predict(X_test)

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

    # KNN supports probability estimates.
    y_probability = model.predict_proba(X_test)[:, 1]

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

    print("\nKNN Optuna V2 Test Results")
    print("--------------------------------")
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

        print("\nApplying Random Oversampling...")

        X_train_ros, y_train_ros = apply_random_oversampling(
            X_train,
            y_train
        )

        print(
            f"Training data after Random Oversampling : "
            f"{X_train_ros.shape}"
        )

        print("\nClass distribution after Random Oversampling:")
        print(y_train_ros.value_counts())

        print("\nStarting Optuna optimization...")
        print(f"Number of trials : {N_TRIALS}")
        print("Cross-validation : 5-fold Stratified CV")
        print("Optimization metric : Class 1 F1-score")

        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=f"sqlite:///{DATABASE_PATH}",
            direction="maximize",
            load_if_exists=False
        )

        study.optimize(
            lambda trial: objective(
                trial,
                X_train_ros,
                y_train_ros
            ),
            n_trials=N_TRIALS
        )

        print("\nOptuna optimization completed.")

        print("\nBest CV F1-score:")
        print(f"{study.best_value:.4f}")

        print("\nBest Parameters:")
        for parameter, value in study.best_params.items():
            print(f"{parameter}: {value}")

        print("\nTraining final KNN model...")

        final_model = train_final_model(
            study.best_params,
            X_train_ros,
            y_train_ros
        )

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        joblib.dump(
            final_model,
            MODEL_PATH
        )

        print(f"\nModel saved to:")
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