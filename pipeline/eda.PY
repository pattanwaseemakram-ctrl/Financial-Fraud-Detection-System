import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset/financial_transactions.csv")

print("Missing Values:")
print(df.isnull().sum())


df.duplicated().sum()


print(df.head())

print(df.tail())

print(df.shape)

print(df.columns)

print(df.info())

print(df.dtypes)

print(df.describe())

for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}")
    print(df[col].unique())

print(df["Type"].value_counts())
print(df["Location"].value_counts())



