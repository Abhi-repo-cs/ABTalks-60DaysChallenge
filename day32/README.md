# Day 32 — Recommendation Systems

## Objective
Build a similarity-based recommendation engine that personalizes product suggestions using customer behavior.

## Dataset
Continues the Day 31 customer dataset containing 300 customers and:
- Customer ID
- Age
- Annual Income
- Spending Score
- Purchase Frequency
- Recency
- Engagement
- Cluster

### Important data note
The Day 31 `customer_clusters.csv` is customer-level data and does not contain actual product transaction columns. Therefore, this project creates a reproducible customer-product interaction matrix from the available behavioral features. For a production recommender, replace this generated matrix with real transaction/click/purchase history.

## Methodology
1. Load customer data.
2. Create customer-product interaction profiles.
3. Calculate customer-to-customer Cosine Similarity.
4. Find the most similar customers.
5. Aggregate similarity-weighted product interactions.
6. Filter products already seen by the target customer.
7. Rank and return Top-N recommendations.

## Evaluation
A holdout evaluation hides a subset of observed interactions and tests whether they appear in the Top-5 recommendations.

Metrics:
- Precision@5
- Recall@5
- Hit Rate@5

## Personalization Strategies
- Customer-similarity personalization
- Similarity-weighted ranking
- Already-purchased filtering
- Top-N recommendations
- Cold-start fallback using popular products

## Files
- `day32.py` — recommendation engine
- `customer_clusters.csv` — Day 31 customer data
- `customer_product_interactions.csv` — customer-product matrix
- `customer_similarity_matrix.csv` — pairwise similarity scores
- `evaluation_metrics.csv` — evaluation results
- `sample_recommendations.csv` — sample recommendation output
- `customer_similarity_distribution.png` — similarity visualization
- `sample_recommendations.png` — recommendation visualization
