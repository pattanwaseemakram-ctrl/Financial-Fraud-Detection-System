import os
import joblib
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
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

from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline


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


# ---------------------------------------------------------
# Custom feature selector
#
# Feature selection is performed only on the training
# portion of each cross-validation fold.
# ---------------------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        threshold=0.01,
        n_estimators=100,
        random_state=42
    ):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):

        # Store input feature names
        self.feature_names_in_ = X.columns.tolist()

        # Create Random Forest for feature importance
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Train only on the current training fold
        self.rf_.fit(X, y)

        # Calculate feature importance
        importance = pd.Series(
            self.rf_.feature_importances_,
            index=self.feature_names_in_
        )

        # Select features above the threshold
        self.selected_features_ = (
            importance[
                importance >= self.threshold
            ].index.tolist()
        )

        # Safety check
        if len(self.selected_features_) == 0:
            self.selected_features_ = [
                importance.idxmax()
            ]

        return self

    def transform(self, X):

        # Keep only features selected from training data
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):

        return self.selected_features_


# ---------------------------------------------------------
# Custom scaler
#
# StandardScaler is fitted only on the training portion
# of each cross-validation fold.
# ---------------------------------------------------------
class SelectiveScaler(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_columns=None):
        self.numerical_columns = numerical_columns

    def fit(self, X, y=None):

        # Scale only numerical columns that remain after
        # feature selection
        self.columns_to_scale_ = [
            column
            for column in self.numerical_columns
            if column in X.columns
        ]

        self.scaler_ = StandardScaler()

        if self.columns_to_scale_:
            self.scaler_.fit(
                X[self.columns_to_scale_]
            )

        return self

    def transform(self, X):

        X = X.copy()

        if self.columns_to_scale_:
            X[self.columns_to_scale_] = (
                self.scaler_.transform(
                    X[self.columns_to_scale_]
                )
            )

        return X


def main():

    try:

        # Create model directory
        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        # ---------------------------------------------------------
        # Load encoded dataset
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
        # Create untouched test set
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
        # Complete end-to-end pipeline
        #
        # Feature Selection
        #       ↓
        # Scaling
        #       ↓
        # ADASYN
        #       ↓
        # XGBoost
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
                "adasyn",
                ADASYN(
                    sampling_strategy=0.5,
                    random_state=42
                )
            ),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

        # ---------------------------------------------------------
        # XGBoost hyperparameter grid
        #
        # We include scale_pos_weight=1 and 10 so that GridSearch
        # can determine whether additional class weighting is
        # useful after ADASYN.
        # ---------------------------------------------------------

        parameter_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__scale_pos_weight": [1, 10]
        }

        # ---------------------------------------------------------
        # 5-fold stratified cross-validation
        # ---------------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        print("\nStarting ADASYN + XGBoost end-to-end fine-tuning...")

        print(
            "\nPipeline:"
            "\nFeature Selection"
            "\n        ↓"
            "\nScaling"
            "\n        ↓"
            "\nADASYN"
            "\n        ↓"
            "\nXGBoost"
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

        # Train
        grid_search.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # Best parameters
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
        # Best pipeline
        # ---------------------------------------------------------

        best_model = grid_search.best_estimator_

        # ---------------------------------------------------------
        # Selected features from final fitted pipeline
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
        # Test prediction
        # ---------------------------------------------------------

        y_pred = best_model.predict(
            X_test
        )

        y_prob = best_model.predict_proba(
            X_test
        )[:, 1]

        # ---------------------------------------------------------
        # Test metrics
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
        # Save complete pipeline
        # ---------------------------------------------------------

        model_path = os.path.join(
            MODEL_DIR,
            "adasyn_xgboost_end_to_end_finetuned.pkl"
        )

        joblib.dump(
            best_model,
            model_path
        )

        print("\nComplete model pipeline saved to:")
        print(model_path)

        print(
            "\nADASYN + XGBoost end-to-end fine-tuning completed successfully."
        )

    except FileNotFoundError as e:

        print("\nError: Required file was not found.")
        print(e)

    except ValueError as e:

        print("\nData processing error:")
        print(e)

    except Exception as e:

        print("\nUnexpected error:")
        print(e)


if __name__ == "__main__":
    main()