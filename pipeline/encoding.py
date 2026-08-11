import pandas as pd

df = pd.read_csv("Dataset/feature_engineered_transactions.csv")
print(df.head())

print("\nCategorical Columns:")
print(df.select_dtypes(include=["object"]).columns)

# Drop unnecessary
df = df.drop(columns=[
    "Transaction ID",
    "Sender Account ID",
    "Receiver Account ID",
    "Timestamp",
    "Device ID",
    "IP Address"
])

print("\nRemaining Columns:")
print(df.columns)


# Check unique values in categorical columns

print("\nType:")
print(df["Type"].unique())

print("\nLocation:")
print(df["Location"].unique())

print("\nTransaction Weekday:")
print(df["Transaction_Weekday"].unique())



# One-Hot Encoding


df = pd.get_dummies(
    df,
    columns=["Type", "Location", "Transaction_Weekday"],
    dtype=int
)

print("\nEncoded Dataset:")
print(df.head())



print("\nEncoded Columns:")
print(df.columns)



# Save the Encoded Dataset
df.to_csv("Dataset/encoded_transactions.csv", index=False)

print("\nEncoded dataset saved successfully.")
