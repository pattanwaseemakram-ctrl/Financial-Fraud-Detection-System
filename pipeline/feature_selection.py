import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# Input file paths
X_TRAIN_FILE = "Dataset/split/X_train.csv"
X_TEST_FILE = "Dataset/split/X_test.csv"

Y_TRAIN_FILE = "Dataset/split/y_train.csv"
Y_TEST_FILE = "Dataset/split/y_test.csv"


# Output file paths
X_TRAIN_SELECTED_FILE = "Dataset/split/X_train_selected.csv"
X_TEST_SELECTED_FILE = "Dataset/split/X_test_selected.csv"

Y_TRAIN_SELECTED_FILE = "Dataset/split/y_train.csv"
Y_TEST_SELECTED_FILE = "Dataset/split/y_test.csv"


# Class containing all feature selection methods
class FeatureSelector:

    # Method to load a dataset
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to get the first five rows
    def get_data_preview(self, df):
        return df.head()

    # Method to get dataset information
    def get_data_info(self, df):
        return df.info()

    # Method to get dataset columns
    def get_columns(self, df):
        return df.columns.tolist()

    # Method to get target distribution
    def get_target_distribution(self, y):
        return y.value_counts()

    # Method to create Random Forest model
    def create_random_forest(self):
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    # Method to train Random Forest only on training data
    def train_random_forest(self, rf, X_train, y_train):
        rf.fit(X_train, y_train)
        return rf

    # Method to calculate feature importance
    def calculate_feature_importance(self, rf, X_train):

        feature_importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": rf.feature_importances_
        })

        return feature_importance

    # Method to sort feature importance
    def sort_feature_importance(self, feature_importance):

        return feature_importance.sort_values(
            by="Importance",
            ascending=False
        )

    # Method to display feature importance graph
    def plot_feature_importance(self, feature_importance):

        plt.figure(figsize=(10, 8))

        plt.barh(
            feature_importance["Feature"],
            feature_importance["Importance"]
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Random Forest Feature Importance")

        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    # Method to select important features
    def select_features(
        self,
        feature_importance,
        threshold=0.01
    ):

        selected_features = feature_importance[
            feature_importance["Importance"] >= threshold
        ]["Feature"].tolist()

        return selected_features

    # Method to create selected training and testing datasets
    def create_selected_datasets(
        self,
        X_train,
        X_test,
        selected_features
    ):

        X_train_selected = X_train[selected_features].copy()
        X_test_selected = X_test[selected_features].copy()

        return X_train_selected, X_test_selected

    # Method to save selected datasets
    def save_selected_datasets(
        self,
        X_train_selected,
        X_test_selected,
        y_train,
        y_test
    ):

        X_train_selected.to_csv(
            X_TRAIN_SELECTED_FILE,
            index=False
        )

        X_test_selected.to_csv(
            X_TEST_SELECTED_FILE,
            index=False
        )

        y_train.to_csv(
            Y_TRAIN_SELECTED_FILE,
            index=False
        )

        y_test.to_csv(
            Y_TEST_SELECTED_FILE,
            index=False
        )

        return True


# Main function to control the feature selection workflow
def main():

    try:

        # Create FeatureSelector object
        selector = FeatureSelector()

        # ---------------------------------------------------------
        # Load already split training and testing data
        # ---------------------------------------------------------

        X_train = selector.load_data(
            X_TRAIN_FILE
        )

        X_test = selector.load_data(
            X_TEST_FILE
        )

        y_train = selector.load_data(
            Y_TRAIN_FILE
        ).squeeze()

        y_test = selector.load_data(
            Y_TEST_FILE
        ).squeeze()

        print("\nTraining data shape:")
        print(X_train.shape)

        print("\nTesting data shape:")
        print(X_test.shape)

        print("\nTraining target shape:")
        print(y_train.shape)

        print("\nTesting target shape:")
        print(y_test.shape)

        # ---------------------------------------------------------
        # Display target distributions
        # ---------------------------------------------------------

        print("\nTraining Target Distribution:")
        print(
            selector.get_target_distribution(
                y_train
            )
        )

        print("\nTesting Target Distribution:")
        print(
            selector.get_target_distribution(
                y_test
            )
        )

        # ---------------------------------------------------------
        # Display dataset columns
        # ---------------------------------------------------------

        print("\nFeature Columns:")
        print(
            selector.get_columns(
                X_train
            )
        )

        # ---------------------------------------------------------
        # Create Random Forest
        # ---------------------------------------------------------

        rf = selector.create_random_forest()

        print(
            "\nRandom Forest model created successfully."
        )

        # ---------------------------------------------------------
        # Train Random Forest ONLY on training data
        #
        # The test data is not used for calculating feature
        # importance.
        # ---------------------------------------------------------

        rf = selector.train_random_forest(
            rf,
            X_train,
            y_train
        )

        print(
            "\nRandom Forest trained using training data only."
        )

        # ---------------------------------------------------------
        # Calculate feature importance
        # ---------------------------------------------------------

        feature_importance = (
            selector.calculate_feature_importance(
                rf,
                X_train
            )
        )

        # Sort feature importance
        feature_importance = (
            selector.sort_feature_importance(
                feature_importance
            )
        )

        print("\nFeature Importance:")
        print(feature_importance)

        # ---------------------------------------------------------
        # Display feature importance graph
        # ---------------------------------------------------------

        selector.plot_feature_importance(
            feature_importance
        )

        # ---------------------------------------------------------
        # Set feature importance threshold
        # ---------------------------------------------------------

        threshold = 0.01

        print(
            f"\nFeature importance threshold: {threshold}"
        )

        # ---------------------------------------------------------
        # Select important features
        # ---------------------------------------------------------

        selected_features = selector.select_features(
            feature_importance,
            threshold
        )

        print("\nSelected Features:")
        print(selected_features)

        print(
            "\nNumber of Selected Features:"
        )
        print(len(selected_features))

        # ---------------------------------------------------------
        # Apply the same selected features to both datasets
        #
        # Important:
        # Features are selected using training data only.
        # The test data only receives the already selected columns.
        # ---------------------------------------------------------

        (
            X_train_selected,
            X_test_selected
        ) = selector.create_selected_datasets(
            X_train,
            X_test,
            selected_features
        )

        print(
            "\nSelected X_train shape:",
            X_train_selected.shape
        )

        print(
            "Selected X_test shape:",
            X_test_selected.shape
        )

        # ---------------------------------------------------------
        # Save selected datasets
        # ---------------------------------------------------------

        save_status = selector.save_selected_datasets(
            X_train_selected,
            X_test_selected,
            y_train,
            y_test
        )

        if save_status:

            print(
                "\nSelected datasets saved successfully."
            )

            print(
                f"\nTraining selected features saved to: "
                f"{X_TRAIN_SELECTED_FILE}"
            )

            print(
                f"Testing selected features saved to: "
                f"{X_TEST_SELECTED_FILE}"
            )

            print(
                f"Training target saved to: "
                f"{Y_TRAIN_SELECTED_FILE}"
            )

            print(
                f"Testing target saved to: "
                f"{Y_TEST_SELECTED_FILE}"
            )

    # Handle missing input files
    except FileNotFoundError:

        print(
            "Error: Required input dataset was not found."
        )

    # Handle invalid data values
    except ValueError as e:

        print(
            "Feature selection data error:",
            e
        )

    # Handle unexpected errors
    except Exception as e:

        print(
            "Unexpected error during feature selection:",
            e
        )


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()