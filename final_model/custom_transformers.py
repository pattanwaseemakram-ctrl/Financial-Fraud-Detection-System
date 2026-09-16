import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


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

        self.feature_names_in_ = X.columns.tolist()

        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.rf_.fit(X, y)

        importance = pd.Series(
            self.rf_.feature_importances_,
            index=self.feature_names_in_
        )

        self.selected_features_ = (
            importance[
                importance >= self.threshold
            ].index.tolist()
        )

        if len(self.selected_features_) == 0:
            self.selected_features_ = [
                importance.idxmax()
            ]

        return self

    def transform(self, X):

        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):

        return self.selected_features_


class SelectiveScaler(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_columns=None):

        self.numerical_columns = numerical_columns

    def fit(self, X, y=None):

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