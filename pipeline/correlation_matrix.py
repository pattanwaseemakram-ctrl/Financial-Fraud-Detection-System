import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load training features
X_train = pd.read_csv("Dataset/X_train.csv")

print("Training Data Shape:")
print(X_train.shape)

# Calculate correlation matrix
correlation_matrix = X_train.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Display correlation heatmap
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

# Find highly correlated feature pairs
upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

print("\nHighly Correlated Feature Pairs:")

for column in upper_triangle.columns:
    for row in upper_triangle.index:

        correlation_value = upper_triangle.loc[row, column]

        if pd.notna(correlation_value) and abs(correlation_value) > 0.90:
            print(
                f"{row} <-> {column}: "
                f"{correlation_value:.3f}"
            )