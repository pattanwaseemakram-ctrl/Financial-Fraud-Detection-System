import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
# sns.countplot(x="Suspicious Activity Flag", data=df)

# plt.title("Class Distribution")
# plt.xlabel("Suspicious Activity Flag")
# plt.ylabel("Count")

# plt.show()