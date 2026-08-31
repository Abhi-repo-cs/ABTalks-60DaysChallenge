# ============================================================
# Day 29 - Customer Segmentation using K-Means Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("Mall_Customers.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 2. Select Numerical Features
# ============================================================

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()

print("\nSelected Features:")
print(X.head())


# ============================================================
# 3. Scale Features
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 4. Experiment with Different Cluster Counts
# ============================================================

inertia = []
silhouette_scores = []

cluster_range = range(2, 11)

for k in cluster_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)

    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )


# ============================================================
# 5. Elbow Method
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    cluster_range,
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.xticks(list(cluster_range))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "elbow_method.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. Silhouette Score
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    cluster_range,
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")
plt.xticks(list(cluster_range))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "silhouette_scores.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. Train Final K-Means Model
# ============================================================

# Select K based on elbow + silhouette analysis
optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ============================================================
# 8. Cluster Summary
# ============================================================

cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster Summary:")
print(cluster_summary.round(2))


# ============================================================
# 9. Customer Count per Cluster
# ============================================================

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nCustomers per Cluster:")
print(cluster_counts)


# ============================================================
# 10. 2D Cluster Visualization
# ============================================================

plt.figure(figsize=(10, 7))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster}",
        s=70
    )


# Plot cluster centers in original scale
centers_scaled = kmeans.cluster_centers_

centers_original = scaler.inverse_transform(
    centers_scaled
)

plt.scatter(
    centers_original[:, 1],
    centers_original[:, 2],
    marker="X",
    s=250,
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "customer_segments.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. Age vs Spending Score
# ============================================================

plt.figure(figsize=(10, 7))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["Age"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster}",
        s=70
    )

plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segments: Age vs Spending Score")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "age_vs_spending.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. Save Segmented Dataset
# ============================================================

df.to_csv(
    "customer_segments.csv",
    index=False
)

print("\nSegmented dataset saved successfully.")


# ============================================================
# 13. Final Silhouette Score
# ============================================================

final_silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print(
    f"\nFinal Silhouette Score: "
    f"{final_silhouette:.3f}"
)