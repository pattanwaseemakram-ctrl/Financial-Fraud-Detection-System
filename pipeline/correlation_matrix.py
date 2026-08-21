import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Input dataset path
INPUT_FILE = "Dataset/X_train.csv"


# Class containing correlation analysis methods
class CorrelationAnalyzer:

    # Method to load training features
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to get dataset shape
    def get_data_shape(self, df):
        return df.shape

    # Method to calculate correlation matrix
    def calculate_correlation(self, df):
        return df.corr()

    # Method to display correlation heatmap
    def display_heatmap(self, correlation_matrix):
        plt.figure(figsize=(14, 10))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

    # Method to find highly correlated feature pairs
    def find_highly_correlated_features(
        self,
        correlation_matrix,
        threshold=0.90
    ):
        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(correlation_matrix.shape),
                k=1
            ).astype(bool))

        highly_correlated_pairs = []

        for column in upper_triangle.columns:
            for row in upper_triangle.index:

                correlation_value = upper_triangle.loc[row, column]

                if (
                    pd.notna(correlation_value)
                    and abs(correlation_value) > threshold
                ):
                    highly_correlated_pairs.append(
                        (
                            row,
                            column,
                            correlation_value))
                    return highly_correlated_pairs


# Main function to control the correlation analysis workflow
def main():

    try:

        # Create a CorrelationAnalyzer object
        analyzer = CorrelationAnalyzer()

        # Load training features
        X_train = analyzer.load_data(INPUT_FILE)

        # Display training data shape
        print("\nTraining Data Shape:")
        print(analyzer.get_data_shape(X_train))

        # Calculate correlation matrix
        correlation_matrix = analyzer.calculate_correlation(
            X_train)

        # Display correlation matrix
        print("\nCorrelation Matrix:")
        print(correlation_matrix)

        # Display correlation heatmap
        analyzer.display_heatmap(correlation_matrix)

        # Find highly correlated feature pairs
        highly_correlated_pairs = (
            analyzer.find_highly_correlated_features(
                correlation_matrix))

        # Display highly correlated feature pairs
        print("\nHighly Correlated Feature Pairs:")

        for row, column, correlation_value in highly_correlated_pairs:
            print(
                f"{row} <-> {column}: "
                f"{correlation_value:.3f}")

    # Handle missing input file
    except FileNotFoundError:
        print("Error: Required input dataset was not found.")

    # Handle missing column
    except KeyError as e:
        print(
            "Error: Required column was not found:",e)

    # Handle invalid correlation data
    except ValueError as e:
        print(
            "Correlation analysis data error:",e)

    # Handle unexpected errors
    except Exception as e:
        print(
            "Unexpected error during correlation analysis:",e)


# Run main() when this file is executed directly
if __name__ == "__main__":
    main()