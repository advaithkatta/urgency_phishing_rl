import pandas as pd

df = pd.read_csv("Enron.csv")

print("Columns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nLabel counts:")
print(df["label"].value_counts())
 