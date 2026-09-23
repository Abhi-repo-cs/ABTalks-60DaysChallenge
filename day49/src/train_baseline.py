from pathlib import Path
import sys
import json
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing import engineer_features, build_preprocessor

DATA_PATH = ROOT / "data" / "raw" / "customer_data.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = engineer_features(df)

target = "churn"
X = df.drop(columns=[target, "customer_id"])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocessor, numeric, categorical = build_preprocessor(X_train)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred, zero_division=0), 4),
    "recall": round(recall_score(y_test, pred, zero_division=0), 4),
    "f1": round(f1_score(y_test, pred, zero_division=0), 4),
    "roc_auc": round(roc_auc_score(y_test, prob), 4),
}

with open(OUTPUT_DIR / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

results = X_test.copy()
results.insert(0, "customer_id", df.loc[X_test.index, "customer_id"])
results["actual_churn"] = y_test
results["predicted_churn"] = pred
results["churn_probability"] = prob.round(4)
results.sort_values("churn_probability", ascending=False).to_csv(
    OUTPUT_DIR / "prediction_outputs.csv", index=False
)

cm = confusion_matrix(y_test, pred)
pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

dump(model, MODEL_DIR / "baseline_churn_pipeline.joblib")

print("Baseline model complete.")
print(json.dumps(metrics, indent=2))
