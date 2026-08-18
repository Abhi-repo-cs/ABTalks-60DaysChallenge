# ============================================================
# DAY 16 - MOVIE RECOMMENDATION USING K-NEAREST NEIGHBORS
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error


# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_csv("movie_ratings_knn.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# ============================================================
# 3. Basic Data Analysis
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumber of Users:", df["User_ID"].nunique())
print("Number of Movies:", df["Movie_ID"].nunique())

print("\nAverage Rating:", round(df["Rating"].mean(), 2))


# ============================================================
# 4. Create Movie-User Rating Matrix
# ============================================================

# Rows = Movies
# Columns = Users
# Values = Ratings

movie_matrix = df.pivot_table(
    index="Movie_Title",
    columns="User_ID",
    values="Rating"
)

print("\nMovie-User Matrix:")
print(movie_matrix.head())


# ============================================================
# 5. Fill Missing Ratings
# ============================================================

# Missing ratings mean the user has not rated that movie.
# We replace them with 0 for similarity calculation.

movie_matrix_filled = movie_matrix.fillna(0)

print("\nMatrix Shape:", movie_matrix_filled.shape)


# ============================================================
# 6. Train KNN Model
# ============================================================

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

knn.fit(movie_matrix_filled)


# ============================================================
# 7. Movie Recommendation Function
# ============================================================

def recommend_movies(movie_name, k=5):

    if movie_name not in movie_matrix_filled.index:
        print("Movie not found in dataset.")
        return

    # Find movie index
    movie_index = movie_matrix_filled.index.get_loc(movie_name)

    # Get movie vector
    movie_vector = movie_matrix_filled.iloc[movie_index].values.reshape(1, -1)

    # Find nearest movies
    distances, indices = knn.kneighbors(
        movie_vector,
        n_neighbors=k + 1
    )

    print(f"\nMovies similar to: {movie_name}")
    print("-" * 50)

    for distance, index in zip(distances[0][1:], indices[0][1:]):

        similar_movie = movie_matrix_filled.index[index]

        similarity = 1 - distance

        print(
            f"{similar_movie:<30} "
            f"Similarity: {similarity:.3f}"
        )


# ============================================================
# 8. Test Recommendation System
# ============================================================

recommend_movies("Inception", k=5)

recommend_movies("Titanic", k=5)

recommend_movies("Toy Story", k=5)


# ============================================================
# 9. Experiment With Different K Values
# ============================================================

k_values = [3, 5, 7, 10, 15]

average_distances = []

for k in k_values:

    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute"
    )

    model.fit(movie_matrix_filled)

    distances, indices = model.kneighbors(
        movie_matrix_filled
    )

    # Ignore the movie itself
    avg_distance = np.mean(distances[:, 1:])

    average_distances.append(avg_distance)

    print(
        f"K = {k} | "
        f"Average Distance = {avg_distance:.4f}"
    )


# ============================================================
# 10. Plot K Comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    average_distances,
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Average Cosine Distance")
plt.title("KNN Performance for Different K Values")

plt.xticks(k_values)
plt.grid(True)

plt.show()


# ============================================================
# 11. Find Best K
# ============================================================

best_index = np.argmin(average_distances)

best_k = k_values[best_index]

print("\nBest K:", best_k)

print(
    "Best K is selected based on the lowest "
    "average cosine distance."
)


# ============================================================
# 12. Final KNN Model
# ============================================================

final_knn = NearestNeighbors(
    n_neighbors=best_k + 1,
    metric="cosine",
    algorithm="brute"
)

final_knn.fit(movie_matrix_filled)


# ============================================================
# 13. Final Recommendation Function
# ============================================================

def final_recommend(movie_name):

    if movie_name not in movie_matrix_filled.index:
        print("Movie not found.")
        return

    movie_index = movie_matrix_filled.index.get_loc(movie_name)

    movie_vector = movie_matrix_filled.iloc[
        movie_index
    ].values.reshape(1, -1)

    distances, indices = final_knn.kneighbors(
        movie_vector
    )

    recommendations = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        movie = movie_matrix_filled.index[index]

        similarity = 1 - distance

        recommendations.append({
            "Movie": movie,
            "Similarity": round(similarity, 3)
        })

    recommendations_df = pd.DataFrame(
        recommendations
    )

    return recommendations_df


# ============================================================
# 14. Generate Final Recommendations
# ============================================================

print("\nFinal Recommendations for Inception:")

print(
    final_recommend("Inception")
)


# ============================================================
# 15. Multiple Movie Recommendations
# ============================================================

movies_to_test = [
    "Inception",
    "Interstellar",
    "The Dark Knight",
    "Titanic",
    "Toy Story"
]

for movie in movies_to_test:

    print("\n" + "=" * 60)
    print(f"Recommendations for {movie}")
    print("=" * 60)

    print(final_recommend(movie))


# ============================================================
# 16. Conclusion
# ============================================================

print("\n" + "=" * 60)
print("PROJECT CONCLUSION")
print("=" * 60)

print(f"""
The K-Nearest Neighbors algorithm was used to build
a movie recommendation system.

The system calculates cosine similarity between movies
based on user rating patterns.

Different K values were tested:
{ k_values }

The selected K value was:
{ best_k }

A smaller K provides more closely related recommendations,
while a larger K considers a broader set of similar movies.

KNN is useful for recommendation systems because it can
identify items that are similar based on historical user
preferences.
""")