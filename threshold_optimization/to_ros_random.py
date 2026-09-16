import os
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler

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


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

# Original encoded dataset
INPUT_FILE = "Dataset/processed data/encoded_transactions.csv"

# Directory for saving threshold results
RESULTS_DIR = "threshold_optimization/results"


# ---------------------------------------------------------
# Numerical columns that require scaling
# ---------------------------------------------------------

NUMERICAL_COLUMNS = [
    "Amount",
    "Account Balance",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month"
]


# ---------------------------------------------------------
# Feature Selection Transformer
#
# Feature selection is performed only on the training
# portion of each cross-validation fold.
# ---------------------------------------------------------

class FeatureSelector(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        threshold=0.01,
        n_estimators=100,
        random_state=42
    ):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(
        self,
        X,
        y
    ):

        # Store the input feature names
        self.feature_names_in_ = X.columns.tolist()

        # Create Random Forest for feature selection
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Train only on the current training fold
        self.rf_.fit(
            X,
            y
        )

        # Calculate feature importance
        importance = pd.Series(
            self.rf_.feature_importances_,
            index=self.feature_names_in_
        )

        # Select features with importance >= 0.01
        self.selected_features_ = (
            importance[
                importance >= self.threshold
            ].index.tolist()
        )

        # Safety check
        if not self.selected_features_:
            self.selected_features_ = [
                importance.idxmax()
            ]

        return self

    def transform(
        self,
        X
    ):

        # Keep only features selected from training data
        return X[
            self.selected_features_
        ].copy()

    def get_feature_names_out(
        self,
        input_features=None
    ):

        return self.selected_features_


# ---------------------------------------------------------
# Selective Scaling Transformer
#
# StandardScaler is fitted only on the current training
# portion of each cross-validation fold.
# ---------------------------------------------------------

class SelectiveScaler(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        numerical_columns=None
    ):
        self.numerical_columns = numerical_columns

    def fit(
        self,
        X,
        y=None
    ):

        # Identify numerical columns that remain after
        # feature selection
        self.columns_to_scale_ = [
            column
            for column in self.numerical_columns
            if column in X.columns
        ]

        # Create StandardScaler
        self.scaler_ = StandardScaler()

        # Fit only on training data
        if self.columns_to_scale_:
            self.scaler_.fit(
                X[
                    self.columns_to_scale_
                ]
            )

        return self

    def transform(
        self,
        X
    ):

        X = X.copy()

        # Transform using the training-fold scaler
        if self.columns_to_scale_:

            X[
                self.columns_to_scale_
            ] = self.scaler_.transform(
                X[
                    self.columns_to_scale_
                ]
            )

        return X


# ---------------------------------------------------------
# Create ROS + Random Forest pipeline
# ---------------------------------------------------------

def create_pipeline():

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
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                max_features="sqrt",
                min_samples_leaf=1,
                min_samples_split=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    return pipeline


def main():

    try:

        print("\nROS + Random Forest Threshold Optimization")
        print("=" * 60)

        # ---------------------------------------------------------
        # Create results directory
        # ---------------------------------------------------------

        os.makedirs(
            RESULTS_DIR,
            exist_ok=True
        )

        # ---------------------------------------------------------
        # Load encoded dataset
        # ---------------------------------------------------------

        df = pd.read_csv(
            INPUT_FILE
        )

        print("\nDataset shape:")
        print(df.shape)

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

        print("\nFeature shape:")
        print(X.shape)

        print("\nTarget distribution:")
        print(y.value_counts())

        # ---------------------------------------------------------
        # Create train/test split
        #
        # Test data remains untouched during threshold selection.
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

        print("Testing data shape:")
        print(X_test.shape)

        print("\nTraining target distribution:")
        print(y_train.value_counts())

        print("\nTesting target distribution:")
        print(y_test.value_counts())

        # ---------------------------------------------------------
        # Create the final clean pipeline
        # ---------------------------------------------------------

        pipeline = create_pipeline()

        print("\nPipeline:")
        print("Feature Selection")
        print("        ↓")
        print("Scaling")
        print("        ↓")
        print("Random Oversampling")
        print("        ↓")
        print("Random Forest")

        print("\nFine-tuned Random Forest parameters:")
        print("n_estimators = 100")
        print("max_depth = 5")
        print("max_features = sqrt")
        print("min_samples_leaf = 1")
        print("min_samples_split = 2")
        print("class_weight = balanced")
        print("ROS sampling_strategy = 0.5")

        # ---------------------------------------------------------
        # Create 5-fold stratified cross-validation
        # ---------------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        # ---------------------------------------------------------
        # Generate out-of-fold probabilities
        #
        # Each training record gets a probability from a model
        # that did not train on that record.
        # ---------------------------------------------------------

        print(
            "\nGenerating out-of-fold probability predictions..."
        )

        y_train_prob = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        print(
            "\nOut-of-fold probability predictions generated successfully."
        )

        # ---------------------------------------------------------
        # Calculate threshold-independent metrics
        # ---------------------------------------------------------

        cv_roc_auc = roc_auc_score(
            y_train,
            y_train_prob
        )

        cv_pr_auc = average_precision_score(
            y_train,
            y_train_prob
        )

        print("\nOut-of-Fold Metrics")
        print("=" * 60)

        print(
            f"ROC-AUC: {cv_roc_auc:.4f}"
        )

        print(
            f"PR-AUC:  {cv_pr_auc:.4f}"
        )

        # ---------------------------------------------------------
        # Thresholds
        #
        # Use a detailed range because we want to understand
        # how the false-positive and recall trade-off changes.
        # ---------------------------------------------------------

        thresholds = [
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.41,
            0.42,
            0.43,
            0.44,
            0.45,
            0.46,
            0.47,
            0.48,
            0.49,
            0.50,
            0.51,
            0.52,
            0.53,
            0.54,
            0.55,
            0.56,
            0.57,
            0.58,
            0.59,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90
        ]

        results = []

        # ---------------------------------------------------------
        # Evaluate every threshold
        # ---------------------------------------------------------

        print("\nThreshold Evaluation")
        print("=" * 120)

        for threshold in thresholds:

            # Convert probabilities into predictions
            y_pred = (
                y_train_prob >= threshold
            ).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(
                y_train,
                y_pred
            )

            precision = precision_score(
                y_train,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_train,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_train,
                y_pred,
                zero_division=0
            )

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                y_train,
                y_pred
            ).ravel()

            # Store results
            results.append({
                "Threshold": threshold,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp
            })

            print(
                f"Threshold: {threshold:.2f} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F1: {f1:.4f} | "
                f"FP: {fp} | "
                f"FN: {fn} | "
                f"TP: {tp}"
            )

        # ---------------------------------------------------------
        # Convert results to DataFrame
        # ---------------------------------------------------------

        results_df = pd.DataFrame(
            results
        )

        # ---------------------------------------------------------
        # Find highest F1 threshold
        #
        # This is a reference only. We will NOT automatically
        # accept it as the final threshold because the false
        # positive count must also be considered.
        # ---------------------------------------------------------

        best_row = results_df.loc[
            results_df["F1"].idxmax()
        ]

        best_threshold = float(
            best_row["Threshold"]
        )

        print("\n")
        print("=" * 60)
        print("HIGHEST-F1 THRESHOLD")
        print("=" * 60)

        print(
            f"Threshold: {best_threshold:.2f}"
        )

        print(
            f"Accuracy:  {best_row['Accuracy']:.4f}"
        )

        print(
            f"Precision: {best_row['Precision']:.4f}"
        )

        print(
            f"Recall:    {best_row['Recall']:.4f}"
        )

        print(
            f"F1 Score:  {best_row['F1']:.4f}"
        )

        print(
            f"TN:        {int(best_row['TN'])}"
        )

        print(
            f"FP:        {int(best_row['FP'])}"
        )

        print(
            f"FN:        {int(best_row['FN'])}"
        )

        print(
            f"TP:        {int(best_row['TP'])}"
        )

        # ---------------------------------------------------------
        # Save threshold comparison
        # ---------------------------------------------------------

        results_file = os.path.join(
            RESULTS_DIR,
            "ros_random_forest_threshold_results.csv"
        )

        results_df.to_csv(
            results_file,
            index=False
        )

        print(
            "\nThreshold results saved to:"
        )

        print(
            results_file
        )

        # ---------------------------------------------------------
        # Do not evaluate the test set yet.
        #
        # We first inspect these threshold results and decide
        # whether the selected threshold is practical.
        # ---------------------------------------------------------

        print(
            "\nThreshold analysis completed."
        )

        print(
            "The test set was kept untouched and was NOT used "
            "for selecting the threshold."
        )

    except FileNotFoundError as e:

        print(
            "\nError: Required file was not found."
        )

        print(e)

    except ValueError as e:

        print(
            "\nData processing error:"
        )

        print(e)

    except Exception as e:

        print(
            "\nUnexpected error:"
        )

        print(e)


if __name__ == "__main__":
    main()