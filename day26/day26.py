# Day 26 - Hyperparameter Tuning
# Optimizing ML Systems with Hyperparameter Tuning

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset Shape:", X.shape)
print("Number of Classes:", y.nunique())

# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. Baseline Model
# --------------------------------------------------

baseline_model = RandomForestClassifier(
    random_state=42
)

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_precision = precision_score(
    y_test,
    baseline_predictions
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions
)

print("\nBaseline Model Performance")
print("--------------------------------")
print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1 Score : {baseline_f1:.4f}")

# --------------------------------------------------
# 4. Hyperparameter Search Space
# --------------------------------------------------

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

# --------------------------------------------------
# 5. RandomizedSearchCV
# --------------------------------------------------

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=30,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

# --------------------------------------------------
# 6. Best Parameters
# --------------------------------------------------

print("\nBest Parameters")
print("--------------------------------")

for parameter, value in random_search.best_params_.items():
    print(f"{parameter}: {value}")

print(
    f"\nBest Cross-Validation F1 Score: "
    f"{random_search.best_score_:.4f}"
)

# --------------------------------------------------
# 7. Tuned Model Evaluation
# --------------------------------------------------

tuned_model = random_search.best_estimator_

tuned_predictions = tuned_model.predict(X_test)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

tuned_precision = precision_score(
    y_test,
    tuned_predictions
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions
)

print("\nTuned Model Performance")
print("--------------------------------")
print(f"Accuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1 Score : {tuned_f1:.4f}")

# --------------------------------------------------
# 8. Performance Comparison
# --------------------------------------------------

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Untuned Model": [
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1
    ],
    "Tuned Model": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1
    ]
})

comparison["Improvement"] = (
    comparison["Tuned Model"]
    - comparison["Untuned Model"]
)

print("\nPerformance Comparison")
print("--------------------------------")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(
    "performance_comparison.csv",
    index=False
)

# --------------------------------------------------
# 9. Save Best Parameters
# --------------------------------------------------

with open("best_parameters.txt", "w") as file:
    file.write("Best Hyperparameters\n")
    file.write("====================\n\n")

    for parameter, value in random_search.best_params_.items():
        file.write(f"{parameter}: {value}\n")

    file.write(
        f"\nBest Cross-Validation F1 Score: "
        f"{random_search.best_score_:.4f}\n"
    )

# --------------------------------------------------
# 10. Visualization
# --------------------------------------------------

metrics = comparison["Metric"]

plt.figure(figsize=(9, 5))

x = range(len(metrics))

plt.bar(
    [i - 0.2 for i in x],
    comparison["Untuned Model"],
    width=0.4,
    label="Untuned"
)

plt.bar(
    [i + 0.2 for i in x],
    comparison["Tuned Model"],
    width=0.4,
    label="Tuned"
)

plt.xticks(list(x), metrics)
plt.ylabel("Score")
plt.title("Untuned vs Tuned Random Forest")
plt.legend()
plt.tight_layout()

plt.savefig(
    "performance_comparison.png",
    dpi=300
)

plt.show()

print("\nOptimization completed successfully.")