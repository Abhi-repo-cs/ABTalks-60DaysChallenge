import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 43
rng = np.random.default_rng(SEED)

# Synthetic customer dataset for a self-contained deployment project.
n = 2500
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 73, n),
    "monthly_charges": rng.uniform(20, 180, n).round(2),
    "total_charges": rng.uniform(50, 12000, n).round(2),
    "support_tickets": rng.poisson(2.2, n),
    "usage_hours": rng.uniform(2, 220, n).round(2),
    "satisfaction_score": rng.uniform(1, 10, n).round(2),
    "contract_type": rng.integers(0, 3, n),  # 0 monthly, 1 annual, 2 two-year
    "payment_delay_days": np.maximum(0, rng.normal(3, 5, n)).round().astype(int)
})

# Realistic nonlinear-ish churn signal with controlled noise.
logit = (
    1.0
    - 0.035 * df.tenure_months
    + 0.010 * df.monthly_charges
    + 0.25 * df.support_tickets
    - 0.012 * df.usage_hours
    - 0.45 * df.satisfaction_score
    - 0.80 * df.contract_type
    + 0.12 * df.payment_delay_days
    + rng.normal(0, 1.0, n)
)
prob = 1 / (1 + np.exp(-logit))
df["churn"] = (rng.random(n) < prob).astype(int)

features = [
    "tenure_months", "monthly_charges", "total_charges",
    "support_tickets", "usage_hours", "satisfaction_score",
    "contract_type", "payment_delay_days"
]
X, y = df[features], df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

model = RandomForestClassifier(
    n_estimators=250, max_depth=9, min_samples_leaf=3,
    class_weight="balanced", random_state=SEED, n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4)
}

os.makedirs("app", exist_ok=True)
joblib.dump(model, "app/customer_churn_model.joblib")
pd.DataFrame([metrics]).to_csv("model_metrics.csv", index=False)

print("Model trained and saved to app/customer_churn_model.joblib")
print(metrics)
