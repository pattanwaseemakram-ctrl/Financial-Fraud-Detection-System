import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Input and output file paths
INPUT_FILE = "Dataset/scaled_transactions.csv"

X_TRAIN_SELECTED_FILE = "Dataset/X_train_selected.csv"
X_TEST_SELECTED_FILE = "Dataset/X_test_selected.csv"
Y_TRAIN_FILE = "Dataset/y_train.csv"
Y_TEST_FILE = "Dataset/y_test.csv"

X_TRAIN_FILE = "Dataset/X_train.csv"
X_TEST_FILE = "Dataset/X_test.csv"


# Class containing all feature selection methods
class FeatureSelector:

    # Method to load the dataset
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

    # Method to separate features and target
    def separate_features_target(self, df):
        X = df.drop("Suspicious Activity Flag", axis=1)
        y = df["Suspicious Activity Flag"]
        return X, y

    # Method to split the dataset into training and testing data
    def split_data(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

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

    # Method to train Random Forest model
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
        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=False
        )
        return feature_importance

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

    # Method to select important features based on threshold
    def select_features(self, feature_importance, threshold=0.01):
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
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
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
            index=False)

        X_test_selected.to_csv(
            X_TEST_SELECTED_FILE,
            index=False)

        y_train.to_csv(
            Y_TRAIN_FILE,
            index=False)

        y_test.to_csv(
            Y_TEST_FILE,
            index=False)
        return True

    # Method to save complete training and testing features
    def save_train_test_features(self, X_train, X_test):
        X_train.to_csv(
            X_TRAIN_FILE,
            index=False)

        X_test.to_csv(
            X_TEST_FILE,
            index=False)
        return True


# Main function to control the feature selection workflow
def main():

    try:

        # Create a FeatureSelector object
        selector = FeatureSelector()

        # Load the scaled dataset
        df = selector.load_data(INPUT_FILE)

        # Display the first five rows
        print("\nFirst 5 Rows:")
        print(selector.get_data_preview(df))

        # Display dataset information
        print("\nDataset Information:")
        selector.get_data_info(df)

        # Display dataset columns
        print("\nDataset Columns:")
        print(selector.get_columns(df))

        # Separate features and target
        X, y = selector.separate_features_target(df)

        # Display features
        print("\nFeatures (X):")
        print(X.columns.tolist())

        # Display target distribution
        print("\nTarget (y):")
        print(selector.get_target_distribution(y))

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = selector.split_data(X, y)

        # Display training and testing shapes
        print("\nX_train:", X_train.shape)
        print("X_test:", X_test.shape)
        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)

        # Display training target distribution
        print("\nTraining Target Distribution:")
        print(selector.get_target_distribution(y_train))

        # Display testing target distribution
        print("\nTesting Target Distribution:")
        print(selector.get_target_distribution(y_test))

        # Create Random Forest model
        rf = selector.create_random_forest()

        # Train Random Forest only on training data
        rf = selector.train_random_forest(
            rf,
            X_train,
            y_train)

        # Calculate feature importance
        feature_importance = selector.calculate_feature_importance(
            rf,
            X_train)

        # Sort feature importance
        feature_importance = selector.sort_feature_importance(
            feature_importance)

        # Display feature importance
        print("\nFeature Importance:")
        print(feature_importance)

        # Display feature importance graph
        selector.plot_feature_importance(feature_importance)

        # Set feature importance threshold
        threshold = 0.01

        # Select important features
        selected_features = selector.select_features(
            feature_importance,
            threshold)

        print("\nSelected Features:")
        print(selected_features)

        print("\nNumber of Selected Features:")
        print(len(selected_features))

        # Create selected training and testing datasets
        X_train_selected, X_test_selected = (
            selector.create_selected_datasets(
                X_train,
                X_test,
                selected_features))

        # Display selected dataset shapes
        print(
            "\nSelected X_train shape:",
            X_train_selected.shape)

        print(
            "Selected X_test shape:",
            X_test_selected.shape)

        # Save selected training and testing datasets
        save_selected_status = selector.save_selected_datasets(
            X_train_selected,
            X_test_selected,
            y_train,
            y_test)

        # Save complete training and testing features
        save_train_test_status = selector.save_train_test_features(
            X_train,
            X_test)

        if save_selected_status and save_train_test_status:
            print("\nSelected datasets saved successfully.")

    # Handle missing input file
    except FileNotFoundError:
        print("Error: Required input dataset was not found.")

    # Handle invalid data values
    except ValueError as e:
        print("Feature selection data error:", e)

    # Handle unexpected errors
    except Exception as e:
        print("Unexpected error during feature selection:", e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()