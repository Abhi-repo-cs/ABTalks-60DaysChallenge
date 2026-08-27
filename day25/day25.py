"""
Day 25 - Cross-Validation Techniques
=====================================
Goal: Evaluate how consistently ML models perform across multiple data
splits using k-fold cross-validation, and compare that against a single
train-test split.

Dataset : Breast Cancer Wisconsin (built into scikit-learn, binary
          classification, 569 samples / 30 features) — used as a stand-in
          for a real production dataset so the script runs anywhere
          without external downloads.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
N_SPLITS = 5

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

# ---------------------------------------------------------------------
# 2. Define candidate models
#    Each is wrapped in a pipeline with scaling since several of these
#    algorithms (SVM, KNN, LogisticRegression) are scale-sensitive.
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE)),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier()),
}

# ---------------------------------------------------------------------
# 3. Single train-test split baseline (what most beginners do first)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

single_split_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    single_split_results[name] = acc

# ---------------------------------------------------------------------
# 4. K-Fold cross-validation (stratified, since this is classification)
# ---------------------------------------------------------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cv_results = {}          # name -> array of per-fold scores
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_results[name] = scores

# ---------------------------------------------------------------------
# 5. Also run plain (non-stratified) K-Fold to show it matters for
#    imbalanced-ish classification problems (stability check).
# ---------------------------------------------------------------------
plain_kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
plain_kfold_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=plain_kfold, scoring="accuracy")
    plain_kfold_results[name] = scores

# ---------------------------------------------------------------------
# 6. Build a comparison table
# ---------------------------------------------------------------------
rows = []
for name in models:
    single_acc = single_split_results[name]
    cv_scores = cv_results[name]
    rows.append({
        "Model": name,
        "Single Split Acc": round(single_acc, 4),
        "CV Mean Acc": round(cv_scores.mean(), 4),
        "CV Std Dev": round(cv_scores.std(), 4),
        "CV Min": round(cv_scores.min(), 4),
        "CV Max": round(cv_scores.max(), 4),
        "Gap (Single - CVmean)": round(single_acc - cv_scores.mean(), 4),
    })

report_df = pd.DataFrame(rows).sort_values("CV Mean Acc", ascending=False)
print("\n=== Performance Comparison: Single Split vs 5-Fold CV ===")
print(report_df.to_string(index=False))

# ---------------------------------------------------------------------
# 7. Stability ranking — lower std dev = more consistent across folds
# ---------------------------------------------------------------------
stability_rank = report_df.sort_values("CV Std Dev")[["Model", "CV Std Dev"]]
print("\n=== Stability Ranking (lower std dev = more consistent) ===")
print(stability_rank.to_string(index=False))

# ---------------------------------------------------------------------
# 8. Save the raw per-fold scores (useful for the written report)
# ---------------------------------------------------------------------
fold_scores_df = pd.DataFrame(cv_results, index=[f"Fold {i+1}" for i in range(N_SPLITS)])
fold_scores_df.to_csv("cv_fold_scores.csv")
report_df.to_csv("performance_comparison.csv", index=False)
print("\nSaved: cv_fold_scores.csv, performance_comparison.csv")

# ---------------------------------------------------------------------
# 9. Visualize: boxplot of CV scores per model (spread = stability)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Boxplot of fold scores
axes[0].boxplot(
    [cv_results[name] for name in models],
    labels=list(models.keys()),
    showmeans=True,
)
axes[0].set_title(f"{N_SPLITS}-Fold CV Accuracy Distribution per Model")
axes[0].set_ylabel("Accuracy")
axes[0].tick_params(axis="x", rotation=30)
axes[0].grid(axis="y", alpha=0.3)

# Bar chart: single split vs CV mean (with error bars = std dev)
names = list(models.keys())
single_vals = [single_split_results[n] for n in names]
cv_means = [cv_results[n].mean() for n in names]
cv_stds = [cv_results[n].std() for n in names]

x_pos = np.arange(len(names))
width = 0.35
axes[1].bar(x_pos - width/2, single_vals, width, label="Single Train-Test Split")
axes[1].bar(x_pos + width/2, cv_means, width, yerr=cv_stds, capsize=4,
            label=f"{N_SPLITS}-Fold CV Mean (± std)")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(names, rotation=30, ha="right")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Single Split vs Cross-Validation Accuracy")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("cv_comparison_plots.png", dpi=150)
print("Saved: cv_comparison_plots.png")

# ---------------------------------------------------------------------
# 10. Reliability observations (auto-generated summary)
# ---------------------------------------------------------------------
most_stable = stability_rank.iloc[0]
least_stable = stability_rank.iloc[-1]
biggest_gap = report_df.reindex(
    report_df["Gap (Single - CVmean)"].abs().sort_values(ascending=False).index
).iloc[0]

print("\n=== Reliability Observations ===")
print(f"- Most stable model across folds: {most_stable['Model']} "
      f"(std dev = {most_stable['CV Std Dev']:.4f})")
print(f"- Least stable model across folds: {least_stable['Model']} "
      f"(std dev = {least_stable['CV Std Dev']:.4f})")
print(f"- Largest gap between single split and CV mean: {biggest_gap['Model']} "
      f"(gap = {biggest_gap['Gap (Single - CVmean)']:.4f}) -> a single split "
      f"can be misleadingly optimistic or pessimistic for this model.")