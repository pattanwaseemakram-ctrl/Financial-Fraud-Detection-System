import pandas as pd
try:
    # Load dataset
    df = pd.read_csv("Dataset/financial_transactions.csv")

    # Missing Values
    print("Missing Values:")
    print(df.isnull().sum())

    # Duplicate Rows
    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nNumerical Columns:")
    print(df.select_dtypes(include=["int64", "float64"]).columns)

    print("\nCategorical Columns:")
    print(df.select_dtypes(include=["object"]).columns)

    # Checking Missing Value Percentage
    print("\nMissing Value Percentage:")
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print(missing_percentage)

    # Checking Duplicate Percentage
    print("\nDuplicate Percentage:")
    duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
    print(f"{duplicate_percentage:.2f}%")

    # Unique Values in Categorical Columns
    print("\nUnique Values in Type:")
    print(df["Type"].unique())

    print("\nUnique Values in Location:")
    print(df["Location"].unique())

    # Checking Invalid Values
    print("\nNegative Amount Values:")
    print((df["Amount"] < 0).sum())

    print("\nNegative Account Balance Values:")
    print((df["Account Balance"] < 0).sum())

    # Converting Timestamp to Datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    print("\nTimestamp Data Type After Conversion:")
    print(df["Timestamp"].dtype)

    # Saving the cleaned dataset

    df.to_csv("Dataset/cleaned_transactions.csv", index=False)

    print("\nCleaned dataset saved successfully!")


except FileNotFoundError:
    print("Error: Input dataset file was not found.")

except ValueError as e:
    print("Data processing error:", e)

except Exception as e:
    print("Unexpected error during preprocessing:", e)