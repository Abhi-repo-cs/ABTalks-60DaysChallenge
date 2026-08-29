# Day 27 - Model Generalization
# Bias-Variance Tradeoff Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset shape:", X.shape)


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 3. Compare Model Complexity
# --------------------------------------------------

depths = [1, 2, 3, 5, 10, 15, 20, None]

results = []

for depth in depths:

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_r2 = r2_score(y_train, train_prediction)
    test_r2 = r2_score(y_test, test_prediction)

    train_mse = mean_squared_error(y_train, train_prediction)
    test_mse = mean_squared_error(y_test, test_prediction)

    results.append({
        "max_depth": depth,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "test_mse": test_mse
    })


results_df = pd.DataFrame(results)

print("\nBias-Variance Analysis")
print(results_df)


# --------------------------------------------------
# 4. Visualize Training vs Validation Performance
# --------------------------------------------------

plt.figure(figsize=(10, 6))

x_labels = [
    str(depth) if depth is not None else "None"
    for depth in depths
]

plt.plot(
    x_labels,
    results_df["train_r2"],
    marker="o",
    label="Training R²"
)

plt.plot(
    x_labels,
    results_df["test_r2"],
    marker="o",
    label="Validation R²"
)

plt.xlabel("Decision Tree Max Depth")
plt.ylabel("R² Score")
plt.title("Training vs Validation Performance")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_vs_validation.png", dpi=300)

plt.show()


# --------------------------------------------------
# 5. Error Comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    x_labels,
    results_df["train_mse"],
    marker="o",
    label="Training MSE"
)

plt.plot(
    x_labels,
    results_df["test_mse"],
    marker="o",
    label="Validation MSE"
)

plt.xlabel("Decision Tree Max Depth")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Validation Error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("error_comparison.png", dpi=300)

plt.show()


# --------------------------------------------------
# 6. Learning Curve
# --------------------------------------------------

best_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

train_sizes, train_scores, validation_scores = learning_curve(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)


plt.figure(figsize=(10, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training Score"
)

plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Validation Score"
)

plt.xlabel("Training Examples")
plt.ylabel("R² Score")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("learning_curve.png", dpi=300)

plt.show()


# --------------------------------------------------
# 7. Final Model Evaluation
# --------------------------------------------------

best_model.fit(X_train, y_train)

final_prediction = best_model.predict(X_test)

print("\nFinal Model Performance")
print("Test R²:", r2_score(y_test, final_prediction))
print("Test MSE:", mean_squared_error(y_test, final_prediction))