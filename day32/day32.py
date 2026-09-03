import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CUSTOMER_FILE = "customer_clusters.csv"

PRODUCTS = [
    "Electronics", "Clothing", "Beauty", "Home & Kitchen",
    "Sports", "Books", "Grocery", "Accessories"
]

def load_customers(path=CUSTOMER_FILE):
    return pd.read_csv(path)

def create_interactions(customers, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for _, r in customers.iterrows():
        weights = np.array([
            r["Spending Score"] / 100 + r["Annual Income"] / 120000,
            r["Spending Score"] / 100,
            (100 - abs(r["Age"] - 30)) / 100,
            r["Engagement"] / 100,
            r["Purchase Frequency"] / 30,
            r["Engagement"] / 100,
            (181 - r["Recency"]) / 180,
            r["Spending Score"] / 100 + r["Engagement"] / 100
        ], dtype=float)

        weights = np.clip(weights, 0.05, None)
        probs = weights / weights.sum()
        rows.append(
            rng.poisson(2 * probs * (r["Purchase Frequency"] / 10 + 1))
        )

    return pd.DataFrame(rows, columns=PRODUCTS, index=customers["Customer ID"]).astype(float)

def build_similarity(interactions):
    return pd.DataFrame(
        cosine_similarity(interactions.to_numpy(dtype=float)),
        index=interactions.index,
        columns=interactions.index
    )

def recommend_products(customer_id, interactions, similarity,
                       top_n=5, n_neighbors=10):
    neighbors = similarity.loc[customer_id].drop(customer_id).nlargest(n_neighbors)
    scores = pd.Series(0.0, index=interactions.columns)

    for neighbor, sim_score in neighbors.items():
        scores += float(max(sim_score, 0)) * interactions.loc[neighbor]

    scores.loc[interactions.loc[customer_id] > 0] = -1
    return scores[scores >= 0].sort_values(ascending=False).head(top_n)

if __name__ == "__main__":
    customers = load_customers()
    interactions = create_interactions(customers)
    similarity = build_similarity(interactions)

    customer_id = interactions.index[0]

    print("Customer:", customer_id)
    print("\nMost Similar Customers:")
    print(similarity.loc[customer_id].drop(customer_id).nlargest(5))

    print("\nTop Recommendations:")
    print(recommend_products(customer_id, interactions, similarity))
