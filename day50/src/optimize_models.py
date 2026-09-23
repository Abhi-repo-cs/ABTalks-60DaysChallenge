from pathlib import Path
import sys
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.preprocessing import engineer_features, build_preprocessor

DATA = ROOT / "data" / "raw" / "customer_data.csv"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
MODEL = ROOT / "models"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
MODEL.mkdir(exist_ok=True)

df = engineer_features(pd.read_csv(DATA))

target = "churn"
X = df.drop(columns=[target, "customer_id"])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preprocessor = build_preprocessor(X_train)

candidates = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1500, class_weight="balanced"),
        {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear", "lbfgs"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 2],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "classifier__n_estimators": [75, 125],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__max_depth": [2, 3],
        },
    ),
}

comparison = []
best_estimators = {}

for name, (estimator, params) in candidates.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    search = GridSearchCV(
        pipe,
        params,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    pred = search.predict(X_test)
    prob = search.predict_proba(X_test)[:, 1]

    train_prob = search.predict_proba(X_train)[:, 1]
    train_pred = search.predict(X_train)

    row = {
        "model": name,
        "cv_best_roc_auc": round(search.best_score_, 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "test_recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "test_f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "test_roc_auc": round(roc_auc_score(y_test, prob), 4),
        "train_roc_auc": round(roc_auc_score(y_train, train_prob), 4),
        "generalization_gap": round(
            roc_auc_score(y_train, train_prob) - roc_auc_score(y_test, prob), 4
        ),
        "best_params": str(search.best_params_),
    }
    comparison.append(row)
    best_estimators[name] = search.best_estimator_

results = pd.DataFrame(comparison).sort_values(
    "test_roc_auc", ascending=False
)
results.to_csv(OUT / "model_comparison.csv", index=False)

# Save the top test-set model for this educational prototype.
best_name = results.iloc[0]["model"]
best_model = best_estimators[best_name]

from joblib import dump
dump(best_model, MODEL / "optimized_model.joblib")

with open(OUT / "optimization_summary.json", "w") as f:
    json.dump(
        {
            "best_model_by_test_roc_auc": best_name,
            "best_model_parameters": results.iloc[0]["best_params"],
        },
        f,
        indent=2,
    )

# Visualization 1: test ROC-AUC comparison.
plt.figure(figsize=(8, 5))
plt.bar(results["model"], results["test_roc_auc"])
plt.ylabel("Test ROC-AUC")
plt.title("Model Comparison — Test ROC-AUC")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(FIG / "model_comparison_roc_auc.png", dpi=150)
plt.close()

# Visualization 2: train/test generalization.
plt.figure(figsize=(8, 5))
x = range(len(results))
width = 0.35
plt.bar([i - width/2 for i in x], results["train_roc_auc"], width, label="Train")
plt.bar([i + width/2 for i in x], results["test_roc_auc"], width, label="Test")
plt.xticks(list(x), results["model"], rotation=15)
plt.ylabel("ROC-AUC")
plt.title("Generalization Check")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "generalization_gap.png", dpi=150)
plt.close()

print("Optimization complete.")
print(results[[
    "model", "cv_best_roc_auc", "test_f1",
    "test_roc_auc", "generalization_gap"
]].to_string(index=False))
