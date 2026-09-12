"""
Day 41 — Predictive Business Analytics
Predictive customer risk system combining segmentation, forecasting-style features,
and churn classification.

Run:
    python day41_risk_prediction.py

Outputs:
    outputs/customer_risk_scores.csv
    outputs/risk_summary.csv
    outputs/risk_dashboard.png
    outputs/business_recommendation_report.md
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

RANDOM_STATE = 42
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------
# 1. Create a reproducible customer-behavior dataset
# ---------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
n = 1200

customer_id = [f"CUST_{i:04d}" for i in range(1, n + 1)]
tenure = rng.integers(1, 61, n)
monthly_spend = np.round(rng.gamma(5, 18, n) + 15, 2)
logins_30d = rng.poisson(10, n)
support_tickets = rng.poisson(1.5, n)
days_since_login = np.clip(rng.normal(12, 9, n).round().astype(int), 0, 60)
payment_failures = rng.poisson(0.25, n)
feature_usage = np.clip(rng.normal(62, 20, n), 5, 100).round(1)
satisfaction = np.clip(rng.normal(7, 1.6, n), 1, 10).round(1)

segment = np.select(
    [
        (monthly_spend >= 100) & (feature_usage >= 65),
        (monthly_spend < 60) & (feature_usage < 55),
        feature_usage >= 75,
    ],
    ["Premium Engaged", "Low Value / Low Engagement", "Power User"],
    default="Standard"
)

# Churn signal generated from interpretable business drivers.
logit = (
    -2.5
    + 0.045 * days_since_login
    + 0.45 * payment_failures
    + 0.20 * support_tickets
    - 0.035 * logins_30d
    - 0.055 * feature_usage
    - 0.20 * satisfaction
    - 0.008 * tenure
    + 0.004 * monthly_spend
)
prob = 1 / (1 + np.exp(-logit))
churn = rng.binomial(1, prob)

df = pd.DataFrame({
    "customer_id": customer_id,
    "tenure_months": tenure,
    "monthly_spend": monthly_spend,
    "logins_30d": logins_30d,
    "support_tickets_30d": support_tickets,
    "days_since_login": days_since_login,
    "payment_failures_90d": payment_failures,
    "feature_usage_pct": feature_usage,
    "satisfaction_score": satisfaction,
    "segment": segment,
    "churned": churn,
})

# ---------------------------------------------------------------------
# 2. Train predictive churn model
# ---------------------------------------------------------------------
features = [
    "tenure_months", "monthly_spend", "logins_30d",
    "support_tickets_30d", "days_since_login",
    "payment_failures_90d", "feature_usage_pct",
    "satisfaction_score", "segment"
]
X = df[features]
y = df["churned"]

numeric = [c for c in features if c != "segment"]
categorical = ["segment"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

test_prob = model.predict_proba(X_test)[:, 1]
pred = (test_prob >= 0.5).astype(int)

print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 3))
print(classification_report(y_test, pred, digits=3))

# Score every customer.
df["churn_probability"] = model.predict_proba(X)[:, 1]

def risk_band(p):
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"

df["risk_band"] = df["churn_probability"].apply(risk_band)
df["risk_rank"] = df["churn_probability"].rank(method="first", ascending=False).astype(int)

def strategy(row):
    if row["risk_band"] == "High":
        if row["days_since_login"] >= 21:
            return "Re-engagement campaign + personalized onboarding"
        if row["payment_failures_90d"] >= 1:
            return "Payment recovery outreach + billing support"
        if row["satisfaction_score"] <= 5:
            return "Priority customer-success call + service recovery"
        return "High-touch retention offer + customer-success outreach"
    if row["risk_band"] == "Medium":
        return "Targeted engagement campaign + feature education"
    return "Maintain engagement with loyalty/value messaging"

df["recommended_retention_strategy"] = df.apply(strategy, axis=1)

ranked = df.sort_values("churn_probability", ascending=False)
ranked.to_csv("outputs/customer_risk_scores.csv", index=False)

summary = (
    df.groupby(["segment", "risk_band"])
      .agg(customers=("customer_id", "count"),
           avg_churn_probability=("churn_probability", "mean"),
           avg_monthly_spend=("monthly_spend", "mean"))
      .reset_index()
)
summary.to_csv("outputs/risk_summary.csv", index=False)

# ---------------------------------------------------------------------
# 3. Dashboard visualization
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(13, 9))

ax1 = plt.subplot(2, 2, 1)
risk_counts = df["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
ax1.bar(risk_counts.index, risk_counts.values)
ax1.set_title("Customers by Risk Band")
ax1.set_ylabel("Customers")

ax2 = plt.subplot(2, 2, 2)
seg_risk = df.groupby("segment")["churn_probability"].mean().sort_values(ascending=False)
ax2.barh(seg_risk.index, seg_risk.values)
ax2.set_title("Average Churn Probability by Segment")
ax2.set_xlabel("Probability")

ax3 = plt.subplot(2, 2, 3)
ax3.scatter(df["days_since_login"], df["churn_probability"], alpha=0.35)
ax3.set_title("Inactivity vs Churn Risk")
ax3.set_xlabel("Days Since Login")
ax3.set_ylabel("Churn Probability")

ax4 = plt.subplot(2, 2, 4)
top10 = ranked.head(10).sort_values("churn_probability")
ax4.barh(top10["customer_id"], top10["churn_probability"])
ax4.set_title("Top 10 Highest-Risk Customers")
ax4.set_xlabel("Churn Probability")

plt.tight_layout()
plt.savefig("outputs/risk_dashboard.png", dpi=180)
plt.close()

# ---------------------------------------------------------------------
# 4. Business recommendation report
# ---------------------------------------------------------------------
high = df[df["risk_band"] == "High"]
medium = df[df["risk_band"] == "Medium"]

report = f"""# Day 41 — Predictive Customer Risk Report

## Executive Summary

This project combines customer behavior, segmentation, and churn signals to create a
predictive customer risk system. Each customer receives a churn probability, risk band,
rank, and recommended retention action.

## Key Results

- Total customers analyzed: **{len(df):,}**
- High-risk customers: **{len(high):,} ({len(high)/len(df):.1%})**
- Medium-risk customers: **{len(medium):,} ({len(medium)/len(df):.1%})**
- Low-risk customers: **{len(df)-len(high)-len(medium):,}**
- Model ROC-AUC on holdout data: **{roc_auc_score(y_test, test_prob):.3f}**

## Recommended Retention Strategy

### High Risk
Prioritize these customers first. Use customer-success outreach, re-engagement journeys,
payment recovery, or service-recovery interventions depending on the dominant signal.

### Medium Risk
Use lower-cost targeted campaigns such as feature education, personalized content,
and engagement reminders.

### Low Risk
Protect retention through loyalty/value messaging and continued product engagement.

## Important Risk Signals

The model uses inactivity, login frequency, feature usage, satisfaction, payment failures,
support activity, tenure, spending, and customer segment. These are predictive signals,
not proof that a customer will churn.

## Next Steps

1. Validate the model on real historical churn data.
2. Calibrate probability thresholds using retention-team capacity and intervention cost.
3. Run an A/B test comparing targeted retention against the current retention process.
4. Track uplift, saved customers, retention revenue, and intervention cost.
5. Retrain regularly as customer behavior and product mix change.

## Files

- `day41_risk_prediction.py` — end-to-end predictive pipeline
- `outputs/customer_risk_scores.csv` — ranked customer risk table
- `outputs/risk_summary.csv` — segment/risk aggregation
- `outputs/risk_dashboard.png` — risk visualization
"""

with open("outputs/business_recommendation_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\nCreated outputs/ with risk scores, summary, dashboard, and report.")
