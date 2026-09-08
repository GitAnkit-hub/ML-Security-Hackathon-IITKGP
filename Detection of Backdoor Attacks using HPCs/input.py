import pandas as pd

df = pd.read_csv("trace.csv")

# print(df.shape)
# print(df.columns)

# print(df.head())

# print(df.describe())

# print(df.isnull().sum())

# what does normla look alike ?


# 01 Visualising the data
#lets make scatter plots
import matplotlib.pyplot as plt

pd.plotting.scatter_matrix(df, figsize=(10,10))
# plt.show()

#A strong relationship between the counters


#02 calculating Cor-relation
# print(df.corr()) # correlation
#The features are not independent.

#03 lets look at covariance also

# print(df.cov())
# Raw covariance is influenced by the units/scales of the features.


#04 Standardized the features

from sklearn.preprocessing import StandardScaler
import numpy as np


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# print(X_scaled[:5])

#we get values measured in terms of standard deviations from the mean.
# fixed the different-units/different-scale problem


# print(X_scaled.shape)
# print(np.cov(X_scaled, rowvar=False))


#05 Calculating the euclidean distance, having mean as (0, 0, 0)
#How far is this clean inference from the average clean inference 
# in standardized feature space?

distance = np.sqrt(np.sum(X_scaled**2, axis = 1))

# print(distance[:10])

# print(distance.max())






# need a way to choose threshold
#06
threshold = 2
# print("Number above threshold:", np.sum(distance > threshold))

# print("Percentage above threshold:", np.mean(distance > threshold) * 100)

#07
# print(np.percentile(distance, [50, 75, 90, 95, 99, 99.5, 100]))




# Euclidean itself can not help
##Distance from the center itself is not enough, direction of the point also matters



#08 #"Can we approximate the normal relationship between these two HPCs with a line?"
x = df["cache-references"].values
y = df["LLC-loads"].values

a, b = np.polyfit(x, y, 1)

# print("slope =", a)
# print("intercept =", b)

predicted = a * x + b
residual = y - predicted

# print("residual std =", np.std(residual))
# print("largest absolute residual =", np.max(np.abs(residual)))





#09 moving towards mahalanbois distance
# taking the shape of the clean distribution into account:
cov = np.cov(X_scaled, rowvar = False)

cov_inv = np.linalg.inv(cov)

# print("Covariance matrix:")
# print(cov)

# print("\nInverse covariance matrix:")
# print(cov_inv)
p = X_scaled[0]
diff = p - X_scaled.mean(axis = 0)

d_squared = diff.T @ cov_inv @ diff

d = np.sqrt(d_squared)

# print("p =", p)
# print("diff =", diff)
# print("Mahalanbois distance =", d)




# calculate for 800 Mahalanbois distances

diff = X_scaled - X_scaled.mean(axis = 0)
mahalanbois_squared = np.einsum(
    'ij, jk, ik -> i',
    diff,
    cov_inv,
    diff
)

mahalanobis_distance = np.sqrt(mahalanbois_squared)

# print(np.percentile(
#     mahalanobis_distance,
#     [50, 75, 90, 95, 99, 99.5, 100]
# ))

# print("min:", mahalanobis_distance.min())
# print("max:", mahalanobis_distance.max())



# 10 Going towards generalization of ML
#training on 600 points
# testing on 200 points

from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(
    X_scaled,
    test_size = 0.25,
    train_size = 0.75,
)

#learn normal behaviour only from the 600 trainign samples

mu = X_train.mean(axis = 0)
cov = np.cov(X_train, rowvar = False)
cov_inv = np.linalg.inv(cov)


# Score training samples
diff_train = X_train - mu

d2_train = np.einsum(
    'ij,jk,ik->i',
    diff_train,
    cov_inv,
    diff_train
)

score_train = np.sqrt(d2_train)

# Score unseen clean samples
diff_val = X_val - mu

d2_val = np.einsum(
    'ij,jk,ik->i',
    diff_val,
    cov_inv,
    diff_val
)

score_val = np.sqrt(d2_val)

# print("TRAIN")
# print(np.percentile(score_train, [50, 75, 90, 95, 99, 100]))

# print("\nVALIDATION (unseen clean)")
# print(np.percentile(score_val, [50, 75, 90, 95, 99, 100]))



# Which traces received the highest Mahalanobis scores?

idx = np.argsort(mahalanobis_distance)[-10:][::-1]

# print(df.iloc[idx])
# print(mahalanobis_distance[idx])


#11 Maximum absolute z-score
z_score = np.max(np.abs(X_scaled), axis=1)

idx = np.argsort(z_score)[-10:][::-1]

print(df.iloc[idx])
print(z_score[idx])
print(mahalanobis_distance[idx])