import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("trace.csv")

X = df.values

# -------------------------
# 1. Split clean data
# -------------------------
X_train, X_val = train_test_split(
    X,
    test_size=0.25,
    random_state=42
)

# -------------------------
# 2. Fit scaler ONLY on training data
# -------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# -------------------------
# 3. Learn clean distribution
# -------------------------
mu = X_train_scaled.mean(axis=0)

cov = np.cov(
    X_train_scaled,
    rowvar=False
)

cov_inv = np.linalg.inv(cov)

# -------------------------
# 4. Function for Mahalanobis score
# -------------------------
def mahalanobis_scores(X, mu, cov_inv):

    diff = X - mu

    squared = np.einsum(
        'ij,jk,ik->i',
        diff,
        cov_inv,
        diff
    )

    return np.sqrt(squared)

# -------------------------
# 5. Scores on training data
# -------------------------
train_scores = mahalanobis_scores(
    X_train_scaled,
    mu,
    cov_inv
)

# -------------------------
# 6. Choose threshold
#    Target ~1% training FPR
# -------------------------
threshold = np.percentile(
    train_scores,
    99
)

# print("Threshold:", threshold)

# -------------------------
# 7. Scores on unseen clean data
# -------------------------
val_scores = mahalanobis_scores(
    X_val_scaled,
    mu,
    cov_inv
)

# -------------------------
# 8. Measure validation FPR
# -------------------------
false_positives = np.sum(
    val_scores > threshold
)

fpr = false_positives / len(val_scores)

# print("False positives:", false_positives)
# print("Validation FPR:", fpr)

# print("\nTraining score percentiles:")
# print(
#     np.percentile(
#         train_scores,
#         [50, 75, 90, 95, 99, 100]
#     )
# )

# print("\nValidation score percentiles:")
# print(
#     np.percentile(
#         val_scores,
#         [50, 75, 90, 95, 99, 100]
#     )
# )






idx = np.where(val_scores > threshold)[0]

print("Indices:", idx)

print("\nFlagged validation traces:")
print(X_val[idx])

print("\nTheir Mahalanobis scores:")
print(val_scores[idx])