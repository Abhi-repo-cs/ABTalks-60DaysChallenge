from pathlib import Path
import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.preprocessing import engineer_features, build_preprocessor

DATA = ROOT / "data" / "raw" / "customer_data.csv"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
MODEL_DIR = ROOT / "models"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

df = engineer_features(pd.read_csv(DATA))
target = "churn"
ids = df["customer_id"]
X = df.drop(columns=[target, "customer_id"])
y = df[target]

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=0.25, random_state=42, stratify=y
)

preprocessor = build_preprocessor(X_train)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    ))
])

params = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_leaf": [1, 2],
}

search = GridSearchCV(
    pipe,
    params,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    refit=True,
)
search.fit(X_train, y_train)

model = search.best_estimator_
test_prob = model.predict_proba(X_test)[:, 1]

# Save fitted model
from joblib import dump
dump(model, MODEL_DIR / "explainable_random_forest_pipeline.joblib")

# Feature names after preprocessing
fitted_preprocessor = model.named_steps["preprocessor"]
rf = model.named_steps["classifier"]
feature_names = fitted_preprocessor.get_feature_names_out()
importance = pd.Series(
    rf.feature_importances_, index=feature_names
).sort_values(ascending=False)

importance_df = importance.rename("importance").reset_index()
importance_df.columns = ["feature", "importance"]
importance_df.to_csv(OUT / "feature_importance.csv", index=False)

# Plot global feature importance
top = importance.head(15).sort_values()
plt.figure(figsize=(9, 6))
plt.barh(top.index, top.values)
plt.xlabel("Importance")
plt.title("Global Feature Importance — Optimized Random Forest")
plt.tight_layout()
plt.savefig(FIG / "global_feature_importance.png", dpi=150)
plt.close()

# SHAP analysis
shap_status = {"available": False, "method": "fallback_feature_importance"}
try:
    import shap
    # TreeExplainer operates on the fitted tree model. Transform a representative sample.
    X_test_transformed = fitted_preprocessor.transform(X_test)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_transformed)

    # Handle binary-class output variants across SHAP versions.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(OUT / "shap_global_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    top_shap = shap_df.head(15).sort_values("mean_abs_shap")
    plt.barh(top_shap["feature"], top_shap["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global SHAP Importance")
    plt.tight_layout()
    plt.savefig(FIG / "shap_global_importance.png", dpi=150)
    plt.close()

    # Per-customer explanation for the first test customer
    idx = 0
    contribution = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv[idx]
    })
    contribution["abs_shap"] = contribution["shap_value"].abs()
    contribution = contribution.sort_values("abs_shap", ascending=False)
    contribution.head(15).to_csv(
        OUT / "customer_001_shap_explanation.csv", index=False
    )

    plt.figure(figsize=(9, 6))
    local = contribution.head(10).sort_values("shap_value")
    plt.barh(local["feature"], local["shap_value"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("SHAP contribution")
    plt.title(f"Local Explanation — Customer {id_test.iloc[idx]}")
    plt.tight_layout()
    plt.savefig(FIG / "local_customer_shap.png", dpi=150)
    plt.close()

    shap_status = {
        "available": True,
        "method": "TreeExplainer",
        "explained_customer": str(id_test.iloc[idx]),
    }
except Exception as exc:
    (OUT / "shap_fallback_reason.txt").write_text(
        f"SHAP analysis was not completed in this environment. "
        f"Fallback feature importance was generated.\nReason: {type(exc).__name__}: {exc}\n",
        encoding="utf-8",
    )

with open(OUT / "explainability_summary.json", "w") as f:
    json.dump({
        "best_parameters": search.best_params_,
        "test_roc_auc": round(roc_auc_score(y_test, test_prob), 4),
        "shap": shap_status,
        "top_features": importance_df.head(10).to_dict(orient="records"),
    }, f, indent=2)

print("Explainable AI analysis complete.")
print("Test ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))
print("Top global features:")
print(importance_df.head(10).to_string(index=False))
print("SHAP:", shap_status)
