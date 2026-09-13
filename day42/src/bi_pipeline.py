import pandas as pd
import numpy as np

def load_data(path="data/customer_intelligence.csv"):
    return pd.read_csv(path)

def calculate_kpis(df):
    return {
        "customers": int(len(df)),
        "revenue": float(df["monthly_revenue"].sum()),
        "avg_revenue": float(df["monthly_revenue"].mean()),
        "avg_engagement": float(df["engagement_score"].mean()),
        "churn_rate": float(df["churned"].mean()),
    }

def add_risk_score(df):
    x = df.copy()
    # Higher score = higher churn risk.
    x["risk_score"] = (
        (100 - x["usage_score"]) * 0.30
        + (100 - x["engagement_score"]) * 0.30
        + x["support_tickets"] * 5 * 0.20
        + x["late_payments"] * 10 * 0.20
    ).clip(0, 100)
    x["risk_band"] = pd.cut(
        x["risk_score"],
        bins=[-1, 30, 60, 100],
        labels=["Low", "Medium", "High"]
    )
    return x

def retention_actions(df):
    actions = []
    for _, r in df.iterrows():
        if r["risk_band"] == "High":
            action = "Immediate retention outreach + usage recovery plan"
        elif r["risk_band"] == "Medium":
            action = "Targeted engagement campaign + customer success check-in"
        else:
            action = "Maintain engagement + expansion opportunity review"
        actions.append(action)
    return actions

def simple_revenue_forecast(df, periods=3):
    # A transparent baseline forecast: use current monthly revenue as a baseline
    # and apply a conservative 2% monthly growth assumption.
    current = df["monthly_revenue"].sum()
    return pd.DataFrame({
        "period": range(1, periods + 1),
        "forecast_revenue": [current * (1.02 ** i) for i in range(1, periods + 1)]
    })
