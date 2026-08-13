import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    df = pd.read_csv("Dataset/encoded_transactions.csv")

    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df["Suspicious Activity Flag"].value_counts())
    print(df["Suspicious Activity Flag"].value_counts(normalize=True) * 100)


    sns.countplot(x="Suspicious Activity Flag", data=df)

    plt.title("Class Distribution")
    plt.xlabel("Suspicious Activity Flag")
    plt.ylabel("Count")

    plt.show()

    print(df.dtypes)
except FileNotFoundError:
    print("Error: Dataset file was not found.")

except KeyError as e:
    print("Error: Target column was not found:", e)

except ValueError as e:
    print("Class imbalance data error:", e)

except Exception as e:
    print("Unexpected error during class imbalance check:", e)


# sns.countplot(x="Suspicious Activity Flag", data=df)

# plt.title("Class Distribution")
# plt.xlabel("Suspicious Activity Flag")
# plt.ylabel("Count")

# plt.show()