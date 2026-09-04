import pandas as pd

df = pd.read_csv("trace.csv")

# print(df.shape)
# print(df.columns)

# print(df.head())

# print(df.describe())

# print(df.isnull().sum())


# what does normla look alike ?
# print(df.corr()) # correlation

# #lets make scatter plots
# import matplotlib.pyplot as plt

# pd.plotting.scatter_matrix(df, figsize=(10,10))
# plt.show()


# lets look at covariance also


print(df.cov())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print(X_scaled[:5])
