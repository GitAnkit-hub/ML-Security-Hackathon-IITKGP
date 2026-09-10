import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------
# Load data
# -------------------------
df = pd.read_csv("trace.csv")

X = df.values


# -------------------------
# Mahalanobis score
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
# Repeat experiment
# -------------------------
results = []

for seed in range(10):

    # 75% train, 25% validation
    X_train, X_val = train_test_split(
        X,
        test_size=0.25,
        random_state=seed
    )

    # -------------------------
    # Fit scaler ONLY on train
    # -------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -------------------------
    # Learn clean distribution
    # -------------------------
    mu = X_train_scaled.mean(axis=0)

    cov = np.cov(
        X_train_scaled,
        rowvar=False
    )

    cov_inv = np.linalg.inv(cov)

    # -------------------------
    # Training scores
    # -------------------------
    train_scores = mahalanobis_scores(
        X_train_scaled,
        mu,
        cov_inv
    )

    # -------------------------
    # Threshold = 99th percentile
    # -------------------------
    threshold = np.percentile(
        train_scores,
        99
    )

    # -------------------------
    # Validation scores
    # -------------------------
    val_scores = mahalanobis_scores(
        X_val_scaled,
        mu,
        cov_inv
    )

    # -------------------------
    # Validation FPR
    # -------------------------
    false_positives = np.sum(
        val_scores > threshold
    )

    fpr = false_positives / len(val_scores)

    results.append(
        [seed, threshold, false_positives, fpr]
    )


# -------------------------
# Display results
# -------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "seed",
        "threshold",
        "false_positives",
        "validation_fpr"
    ]
)

print(results_df)

print("\nAverage FPR:",
      results_df["validation_fpr"].mean())

print("Std FPR:",
      results_df["validation_fpr"].std())

print("\nAverage threshold:",
      results_df["threshold"].mean())

print("Std threshold:",
      results_df["threshold"].std())


#Outcome:
#A threshold derived from clean training data at the 99th percentile produces roughly 1% false alarms on unseen clean data.