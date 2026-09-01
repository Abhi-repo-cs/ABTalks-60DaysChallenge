# ============================================================
# Day 30 - Clustering Optimization
# Finding the Ideal Number of Customer Segments
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

DATA_PATH = "customer_segments.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DAY 30 - CLUSTERING OPTIMIZATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. Check Dataset Quality
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 3. Select Clustering Features
# ============================================================

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Purchases_Per_Month",
    "Website_Visits_Per_Month"
]

X = df[features].copy()

print("\nFeatures Used for Clustering:")
for feature in features:
    print("-", feature)


# ============================================================
# 4. Standardize Features
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized successfully.")


# ============================================================
# 5. Test Multiple Values of K
# ============================================================

k_values = range(2, 11)

inertia_values = []
silhouette_values = []

print("\n" + "=" * 60)
print("CLUSTERING QUALITY ANALYSIS")
print("=" * 60)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    inertia_values.append(inertia)
    silhouette_values.append(silhouette)

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:8.2f} | "
        f"Silhouette Score = {silhouette:.4f}"
    )


# ============================================================
# 6. Create Evaluation DataFrame
# ============================================================

evaluation_df = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertia_values,
    "Silhouette Score": silhouette_values
})

print("\nClustering Evaluation:")
print(evaluation_df.round(4))


# ============================================================
# 7. Identify Best K Using Silhouette Score
# ============================================================

best_silhouette_index = np.argmax(
    silhouette_values
)

best_k_silhouette = list(k_values)[
    best_silhouette_index
]

best_silhouette = silhouette_values[
    best_silhouette_index
]

print("\nBest K according to Silhouette Score:")
print(f"K = {best_k_silhouette}")
print(f"Silhouette Score = {best_silhouette:.4f}")


# ============================================================
# 8. Elbow Method
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Clustering Optimization")

plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "day30_elbow_method.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. Silhouette Score Visualization
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    silhouette_values,
    marker="o"
)

plt.axvline(
    best_k_silhouette,
    linestyle="--",
    label=f"Best K = {best_k_silhouette}"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")

plt.xticks(list(k_values))
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "day30_silhouette_scores.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. Compare K Values
# ============================================================

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(
    k_values,
    inertia_values,
    marker="o",
    label="Inertia"
)

ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia")

ax2 = ax1.twinx()

ax2.plot(
    k_values,
    silhouette_values,
    marker="s",
    label="Silhouette Score"
)

ax2.set_ylabel("Silhouette Score")

plt.title(
    "Clustering Improvement: Inertia vs Silhouette Score"
)

ax1.grid(True)

plt.tight_layout()

plt.savefig(
    "day30_clustering_quality.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. Select Final K
# ============================================================

# Use silhouette score as the primary quantitative criterion.
# The Elbow Method is used as supporting evidence.

optimal_k = best_k_silhouette

print("\n" + "=" * 60)
print("FINAL CLUSTERING MODEL")
print("=" * 60)

print(f"Selected K: {optimal_k}")
print(f"Silhouette Score: {best_silhouette:.4f}")


# ============================================================
# 12. Train Final K-Means Model
# ============================================================

final_kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = final_kmeans.fit_predict(
    X_scaled
)


# ============================================================
# 13. Cluster Size Analysis
# ============================================================

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nCustomers in Each Cluster:")
print(cluster_sizes)


# ============================================================
# 14. Cluster Profile
# ============================================================

cluster_profile = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

print("\n" + "=" * 60)
print("CUSTOMER SEGMENT PROFILES")
print("=" * 60)

print(cluster_profile)


# ============================================================
# 15. Add Customer Counts
# ============================================================

cluster_profile["Customer_Count"] = (
    cluster_sizes
)

cluster_profile["Percentage"] = (
    cluster_sizes / len(df) * 100
).round(2)

print("\nComplete Cluster Profile:")
print(cluster_profile)


# ============================================================
# 16. Visualize Final Customer Segments
# ============================================================

plt.figure(figsize=(10, 7))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster}",
        s=70
    )


# Convert centroids back to original scale
centroids_scaled = (
    final_kmeans.cluster_centers_
)

centroids_original = (
    scaler.inverse_transform(
        centroids_scaled
    )
)

plt.scatter(
    centroids_original[:, 1],
    centroids_original[:, 2],
    marker="X",
    s=250,
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(
    f"Optimized Customer Segmentation (K={optimal_k})"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "day30_optimized_segments.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. Save Results
# ============================================================

evaluation_df.to_csv(
    "day30_clustering_evaluation.csv",
    index=False
)

df.to_csv(
    "day30_customer_segments.csv",
    index=False
)

cluster_profile.to_csv(
    "day30_segment_profiles.csv"
)


# ============================================================
# 18. Generate Optimization Summary
# ============================================================

print("\n" + "=" * 60)
print("OPTIMIZATION SUMMARY")
print("=" * 60)

print(f"""
Number of customers       : {len(df)}
Features used             : {len(features)}
K values tested           : 2 to 10
Optimal K                 : {optimal_k}
Best Silhouette Score     : {best_silhouette:.4f}

The final clustering model was selected by comparing
the Elbow Method and Silhouette Scores across multiple
cluster counts.

The resulting customer segments can be used to support
targeted marketing, customer retention, and personalized
business strategies.
""")


# ============================================================
# 19. Business-Oriented Cluster Interpretation
# ============================================================

print("=" * 60)
print("BUSINESS SEGMENTATION STRATEGY")
print("=" * 60)

for cluster in cluster_profile.index:

    profile = cluster_profile.loc[cluster]

    income = profile["Annual Income (k$)"]
    spending = profile["Spending Score (1-100)"]

    if income >= 70 and spending >= 60:

        segment_type = "High-Value Customers"

        strategy = (
            "Prioritize retention, VIP rewards, "
            "premium products, and exclusive offers."
        )

    elif income >= 70 and spending < 60:

        segment_type = "High-Income Potential Customers"

        strategy = (
            "Use personalized recommendations, "
            "targeted promotions, and engagement campaigns."
        )

    elif income < 50 and spending >= 60:

        segment_type = "High-Engagement Budget Customers"

        strategy = (
            "Use loyalty rewards, affordable bundles, "
            "discounts, and frequent-purchase incentives."
        )

    elif income < 50 and spending < 40:

        segment_type = "Low-Engagement Customers"

        strategy = (
            "Use cost-efficient campaigns and "
            "entry-level products."
        )

    else:

        segment_type = "Mid-Value Customers"

        strategy = (
            "Focus on cross-selling, loyalty programs, "
            "and increasing purchase frequency."
        )

    print(f"\nCluster {cluster}")
    print(f"Type     : {segment_type}")
    print(f"Customers: {int(profile['Customer_Count'])}")
    print(f"Strategy : {strategy}")


# ============================================================
# END
# ============================================================

print("\nDay 30 clustering optimization completed successfully.")