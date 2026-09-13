import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.bi_pipeline import load_data, calculate_kpis, add_risk_score, retention_actions, simple_revenue_forecast

st.set_page_config(page_title="Day 42 Integrated BI", layout="wide")
st.title("Day 42 — Integrated Customer Intelligence BI")
st.caption("Forecasting • KPI Tracking • Retention Analytics • Predictive Risk")

df = load_data()
scored = add_risk_score(df)
scored["retention_action"] = retention_actions(scored)
kpis = calculate_kpis(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{kpis['customers']:,}")
c2.metric("Monthly Revenue", f"${kpis['revenue']:,.0f}")
c3.metric("Avg Engagement", f"{kpis['avg_engagement']:.1f}")
c4.metric("Churn Rate", f"{kpis['churn_rate']:.1%}")

st.subheader("Revenue Forecast")
forecast = simple_revenue_forecast(df, 6)
fig = px.line(forecast, x="period", y="forecast_revenue", markers=True,
              labels={"period":"Future Month", "forecast_revenue":"Forecast Revenue"})
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Customer Risk Distribution")
    risk_counts = scored["risk_band"].value_counts().reset_index()
    risk_counts.columns = ["risk_band", "customers"]
    st.plotly_chart(px.bar(risk_counts, x="risk_band", y="customers"),
                    use_container_width=True)
with right:
    st.subheader("Risk vs Revenue")
    st.plotly_chart(px.scatter(scored, x="risk_score", y="monthly_revenue",
                               color="risk_band", hover_name="customer_id",
                               labels={"monthly_revenue":"Monthly Revenue",
                                       "risk_score":"Risk Score"}),
                    use_container_width=True)

st.subheader("Priority Retention Queue")
st.dataframe(
    scored.sort_values(["risk_score", "monthly_revenue"], ascending=[False, False])[
        ["customer_id","segment","monthly_revenue","risk_score","risk_band","retention_action"]
    ],
    use_container_width=True
)

st.info("Business rule: combine risk with customer value. A high-risk, high-revenue customer should receive the fastest intervention.")
