import os
import sys
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
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

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# ---------------------------------------------------------
# Add final_model directory to Python path
# ---------------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "final_model"
        )
    )
)

from final_model.custom_transformers import (
    FeatureSelector,
    SelectiveScaler
)


# Input dataset
INPUT_FILE = (
    "Dataset/processed data/encoded_transactions_improved.csv"
)

# Output directory
MODEL_DIR = "improved_model/models"


# Numerical columns that require scaling
NUMERICAL_COLUMNS = [
    "Amount",
    "Account Balance",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Amount_to_Balance_Ratio",
    "Amount_Percent_of_Balance",
    "Spending_Deviation_Amount"
]


def main():

    try:

        # ---------------------------------------------------------
        # Create output directory
        # ---------------------------------------------------------

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        # ---------------------------------------------------------
        # Load improved encoded dataset
        # ---------------------------------------------------------

        df = pd.read_csv(
            INPUT_FILE
        )

        print("Improved encoded dataset loaded successfully.")

        print("\nDataset shape:")
        print(df.shape)

        # ---------------------------------------------------------
        # Separate features and target
        # ---------------------------------------------------------

        target_column = "Suspicious Activity Flag"

        if target_column not in df.columns:

            raise ValueError(
                f"Target column '{target_column}' was not found."
            )

        X = df.drop(
            target_column,
            axis=1
        )

        y = df[
            target_column
        ]

        print("\nFeature shape:")
        print(X.shape)

        print("\nTarget distribution:")
        print(y.value_counts())

        # ---------------------------------------------------------
        # Create final train/test split
        #
        # The test data remains untouched during cross-validation
        # and model selection.
        # ---------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print("\nTraining data shape:")
        print(X_train.shape)

        print("\nTesting data shape:")
        print(X_test.shape)

        print("\nTraining target distribution:")
        print(y_train.value_counts())

        print("\nTesting target distribution:")
        print(y_test.value_counts())

        # ---------------------------------------------------------
        # Create complete end-to-end pipeline
        #
        # Feature Selection
        #       ↓
        # Scaling
        #       ↓
        # Random OverSampling
        #       ↓
        # Logistic Regression
        #
        # Feature selection and scaling are performed within
        # each CV training fold.
        # ---------------------------------------------------------

        pipeline = Pipeline([
            (
                "feature_selection",
                FeatureSelector(
                    threshold=0.01,
                    n_estimators=100,
                    random_state=42
                )
            ),
            (
                "scaler",
                SelectiveScaler(
                    numerical_columns=NUMERICAL_COLUMNS
                )
            ),
            (
                "ros",
                RandomOverSampler(
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

        # ---------------------------------------------------------
        # Hyperparameter grid
        # ---------------------------------------------------------

        parameter_grid = {
            "model__C": [
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                5,
                10
            ],
            "model__solver": [
                "lbfgs",
                "liblinear"
            ],
            "model__class_weight": [
                None,
                "balanced"
            ]
        }

        # ---------------------------------------------------------
        # 5-fold stratified cross-validation
        # ---------------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        print("\nStarting improved end-to-end fine-tuning...")

        print(
            "\nPipeline:"
            "\nFeature Selection"
            "\n        ↓"
            "\nScaling"
            "\n        ↓"
            "\nRandom OverSampling"
            "\n        ↓"
            "\nLogistic Regression"
        )

        print(
            "\nNumber of hyperparameter combinations:"
        )

        print(
            11 * 2 * 2
        )

        # ---------------------------------------------------------
        # GridSearchCV
        # ---------------------------------------------------------

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        # ---------------------------------------------------------
        # Train model
        # ---------------------------------------------------------

        grid_search.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # Display best parameters
        # ---------------------------------------------------------

        print("\nBest Parameters:")
        print(
            grid_search.best_params_
        )

        print("\nBest Cross-Validation F1 Score:")
        print(
            f"{grid_search.best_score_:.4f}"
        )

        # ---------------------------------------------------------
        # Get best complete pipeline
        # ---------------------------------------------------------

        best_model = grid_search.best_estimator_

        # ---------------------------------------------------------
        # Display selected features
        # ---------------------------------------------------------

        selected_features = (
            best_model
            .named_steps["feature_selection"]
            .selected_features_
        )

        print("\nSelected Features:")
        print(
            selected_features
        )

        print("\nNumber of Selected Features:")
        print(
            len(selected_features)
        )

        # ---------------------------------------------------------
        # Generate predictions on untouched test data
        # ---------------------------------------------------------

        y_pred = best_model.predict(
            X_test
        )

        # Generate probability scores
        y_prob = best_model.predict_proba(
            X_test
        )[:, 1]

        # ---------------------------------------------------------
        # Calculate test metrics
        # ---------------------------------------------------------

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
            y_prob
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        # ---------------------------------------------------------
        # Confusion matrix
        # ---------------------------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()

        # ---------------------------------------------------------
        # Display test results
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("IMPROVED MODEL TEST RESULTS")
        print("=" * 60)

        print("\nModel:")
        print(
            "Improved ROS + Logistic Regression"
        )

        print(
            f"\nAccuracy:  {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"F1 Score:  {f1:.4f}"
        )

        print(
            f"ROC-AUC:   {roc_auc:.4f}"
        )

        print(
            f"PR-AUC:    {pr_auc:.4f}"
        )

        # ---------------------------------------------------------
        # Display confusion matrix
        # ---------------------------------------------------------

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )

        print("\nConfusion Matrix Values:")

        print(
            f"True Negatives:  {tn}"
        )

        print(
            f"False Positives: {fp}"
        )

        print(
            f"False Negatives: {fn}"
        )

        print(
            f"True Positives:  {tp}"
        )

        # ---------------------------------------------------------
        # Classification report
        # ---------------------------------------------------------

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        # ---------------------------------------------------------
        # Save improved model
        # ---------------------------------------------------------

        model_path = os.path.join(
            MODEL_DIR,
            "improved_ros_logistic.pkl"
        )

        joblib.dump(
            best_model,
            model_path
        )

        print("\nImproved model saved to:")
        print(model_path)

        print(
            "\nImproved ROS + Logistic Regression "
            "training completed successfully."
        )

    # Handle missing files
    except FileNotFoundError as e:

        print(
            "\nError: Required file was not found."
        )

        print(e)

    # Handle invalid data
    except ValueError as e:

        print(
            "\nData processing error:"
        )

        print(e)

    # Handle unexpected errors
    except Exception as e:

        print(
            "\nUnexpected error:"
        )

        print(e)


if __name__ == "__main__":
    main()