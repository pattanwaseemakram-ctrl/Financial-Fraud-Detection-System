import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    # Load dataset
    df = pd.read_csv("Dataset/feature_engineered_transactions.csv")

    print("Numerical Columns:")
    print(df.select_dtypes(include=["int64", "float64"]).columns)

    # Box Plot for Transaction Amount
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Amount"])

    plt.title("Box Plot of Transaction Amount")
    plt.xlabel("Amount")

    plt.show()



 # IQR Method

    Q1 = df["Amount"].quantile(0.25)
    Q3 = df["Amount"].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)

   # Detect Outliers
    outliers = df[(df["Amount"] < lower_limit) | (df["Amount"] > upper_limit)]

    print("\nNumber of Outliers:", len(outliers))

   # Display first 5 outliers
    print(outliers.head())


   # Box Plot for Account Balance

    plt.figure(figsize=(8,5))
    sns.boxplot(x=df["Account Balance"])

    plt.title("Box Plot of Account Balance")
    plt.xlabel("Account Balance")

    plt.show()


    #Detection for Account Balance

    Q1_balance = df["Account Balance"].quantile(0.25)
    Q3_balance = df["Account Balance"].quantile(0.75)

    IQR_balance = Q3_balance - Q1_balance

    lower_limit_balance = Q1_balance - (1.5 * IQR_balance)
    upper_limit_balance = Q3_balance + (1.5 * IQR_balance)

    print("\n---------- Account Balance ----------")
    print("Q1:", Q1_balance)
    print("Q3:", Q3_balance)
    print("IQR:", IQR_balance)
    print("Lower Limit:", lower_limit_balance)
    print("Upper Limit:", upper_limit_balance)

   # Detect Outliers for Account Balance
    outliers_balance = df[
    (df["Account Balance"] < lower_limit_balance) |
    (df["Account Balance"] > upper_limit_balance)
 ]

    print("\nNumber of Account Balance Outliers:", len(outliers_balance))

    print(outliers_balance.head())
except FileNotFoundError:
    print("Error: Dataset file was not found.")

except KeyError as e:
    print("Error: Required column was not found:", e)

except ValueError as e:
    print("Outlier detection data error:", e)

except Exception as e:
    print("Unexpected error during outlier detection:", e)