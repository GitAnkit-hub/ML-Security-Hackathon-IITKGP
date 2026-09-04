import pandas as pd

df = pd.read_csv("trace.csv")

print(df.shape)
print(df.columns)

print(df.head())

print(df.describe())

print(df.isnull().sum())