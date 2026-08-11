import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Dataset/encoded_transactions.csv")

print(df.head())

print("\nDataset Information:")
print(df.info())



print("\nColumns Before Scaling:")
print(df.columns)

# Continuous numerical columns to scale
numerical_columns = [
    "Amount",
    "Account Balance",
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month"
]

print("\nColumns Selected for Scaling:")
print(numerical_columns)



# Create StandardScaler 
scaler = StandardScaler()

print("\nStandardScaler object created successfully.")


# Apply StandardScaler to the selected columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nFeature Scaling Applied Successfully.")




print("\nScaled Dataset:")
print(df.head())




print("\nMean of Scaled Columns:")
print(df[numerical_columns].mean())

print("\nStandard Deviation of Scaled Columns:")
print(df[numerical_columns].std())




# Save the scaled dataset
df.to_csv("Dataset/scaled_transactions.csv", index=False)

print("\nScaled dataset saved successfully.")