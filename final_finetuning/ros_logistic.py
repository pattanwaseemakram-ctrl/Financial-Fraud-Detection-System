import os
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

from final_model.custom_transformers import (
    FeatureSelector,
    SelectiveScaler
)


# Input dataset
INPUT_FILE = "Dataset/processed data/encoded_transactions.csv"

# Output directory
MODEL_DIR = "final_finetuning/models"

# Numerical columns that require scaling
NUMERICAL_COLUMNS = [
    "Amount",
    "Account Balance",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month"
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
        # Load encoded dataset
        #
        # We start from the encoded dataset instead of the old
        # scaled dataset so that scaling happens correctly inside
        # cross-validation.
        # ---------------------------------------------------------

        df = pd.read_csv(INPUT_FILE)

        print("Dataset shape:", df.shape)

        # ---------------------------------------------------------
        # Separate features and target
        # ---------------------------------------------------------

        X = df.drop(
            "Suspicious Activity Flag",
            axis=1
        )

        y = df[
            "Suspicious Activity Flag"
        ]

        print("\nFeature shape:", X.shape)

        print("\nTarget distribution:")
        print(y.value_counts())

        # ---------------------------------------------------------
        # Create the final train/test split
        #
        # The test set is kept completely untouched.
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
        # Create complete pipeline
        #
        # Order:
        #
        # Feature Selection
        #       ↓
        # Scaling
        #       ↓
        # Random Oversampling
        #       ↓
        # Logistic Regression
        #
        # All steps are fitted separately inside each CV fold.
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
        #
        # The same Logistic Regression search used previously
        # is retained for a fair comparison.
        # ---------------------------------------------------------

        parameter_grid = {
            "model__C": [
                0.001,
                0.01,
                0.1,
                1,
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

        print("\nStarting end-to-end fine-tuning...")
        print(
            "\nPipeline:"
            "\nFeature Selection"
            "\n        ↓"
            "\nScaling"
            "\n        ↓"
            "\nRandom Oversampling"
            "\n        ↓"
            "\nLogistic Regression"
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
        # Train the complete pipeline
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
        # Display selected features from the final fitted model
        #
        # This is the feature set selected when the best pipeline
        # was refitted on the complete training data.
        # ---------------------------------------------------------

        selected_features = (
            best_model
            .named_steps["feature_selection"]
            .selected_features_
        )

        print("\nSelected Features:")
        print(selected_features)

        print("\nNumber of Selected Features:")
        print(len(selected_features))

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
        print("FINAL TEST RESULTS")
        print("=" * 60)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")

        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )

        print("\nConfusion Matrix Values:")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives:  {tp}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        # ---------------------------------------------------------
        # Save the complete best pipeline
        # ---------------------------------------------------------

        model_path = os.path.join(
            MODEL_DIR,
            "ros_logistic_end_to_end_finetuned.pkl"
        )

        joblib.dump(
            best_model,
            model_path
        )

        print("\nComplete model pipeline saved to:")
        print(model_path)

        print(
            "\nEnd-to-end fine-tuning completed successfully."
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