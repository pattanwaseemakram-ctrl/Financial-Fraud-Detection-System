import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.ensemble import RandomForestClassifier

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
#
# This allows the script to import custom_transformers.py.
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

from final_model.custom_transformers import( 
    FeatureSelector,
    SelectiveScaler
)


# ---------------------------------------------------------
# Input dataset
# ---------------------------------------------------------

INPUT_FILE = (
    "Dataset/processed data/encoded_transactions_improved.csv"
)

# Output directory
MODEL_DIR = "improved_random_forest/models"


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

        print("Loading improved encoded dataset...")

        df = pd.read_csv(
            INPUT_FILE
        )

        print(
            "Improved encoded dataset loaded successfully."
        )

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
        print(
            y.value_counts()
        )

        # ---------------------------------------------------------
        # Create train/test split
        #
        # Test data remains untouched during model selection.
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
        print(
            y_train.value_counts()
        )

        print("\nTesting target distribution:")
        print(
            y_test.value_counts()
        )

        # ---------------------------------------------------------
        # Create Random Forest model
        # ---------------------------------------------------------

        random_forest = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

        # ---------------------------------------------------------
        # Create complete pipeline
        #
        # Feature Selection
        #       ↓
        # Random OverSampling
        #       ↓
        # Random Forest
        #
        # Feature selection and ROS are performed inside
        # each cross-validation training fold.
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
                "ros",
                RandomOverSampler(
                    sampling_strategy=0.5,
                    random_state=42
                )
            ),
            (
                "model",
                random_forest
            )
        ])

        # ---------------------------------------------------------
        # Reduced Random Forest hyperparameter grid
        #
        # The previous grid contained 432 configurations.
        # This targeted grid contains only 24 configurations.
        # ---------------------------------------------------------

        parameter_grid = {
            "model__n_estimators": [
                100,
                200
            ],
            "model__max_depth": [
                5,
                10
            ],
            "model__min_samples_split": [
                2,
                5
            ],
            "model__min_samples_leaf": [
                1
            ],
            "model__max_features": [
                "sqrt"
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

        print("\nStarting improved ROS + Random Forest fine-tuning...")

        print(
            "\nPipeline:"
            "\nFeature Selection"
            "\n        ↓"
            "\nRandom OverSampling"
            "\n        ↓"
            "\nRandom Forest"
        )

        print(
            "\nNumber of hyperparameter combinations:"
        )

        print(
            2 * 2 * 2 * 1 * 1 * 2
        )

        print(
            "\nTotal cross-validation fits:"
        )

        print(
            24 * 5
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
        print("IMPROVED ROS + RANDOM FOREST TEST RESULTS")
        print("=" * 60)

        print(
            "\nModel:"
            "\nImproved Features + ROS + Random Forest"
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
        # Save improved Random Forest model
        # ---------------------------------------------------------

        model_path = os.path.join(
            MODEL_DIR,
            "improved_ros_random_forest.pkl"
        )

        joblib.dump(
            best_model,
            model_path
        )

        print("\nImproved Random Forest model saved to:")
        print(
            model_path
        )

        print(
            "\nImproved ROS + Random Forest training "
            "completed successfully."
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