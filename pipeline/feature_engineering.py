import pandas as pd

# Load Dataset
df = pd.read_csv("Dataset/cleaned_transactions.csv")

# Convert Timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Feature Engineering
df["Transaction_Hour"] = df["Timestamp"].dt.hour
df["Transaction_Day"] = df["Timestamp"].dt.day
df["Transaction_Month"] = df["Timestamp"].dt.month
df["Transaction_Weekday"] = df["Timestamp"].dt.day_name()

df["High_Value_Transaction"] = (df["Amount"] > 10000).astype(int)
df["Low_Balance"] = (df["Account Balance"] < 1000).astype(int)

# Display dataset
print(df.head())

# Save engineered dataset
df.to_csv("Dataset/feature_engineered_transactions.csv", index=False)

print("Feature Engineering Completed Successfully.")