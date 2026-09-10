import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Business KPI Analytics", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("business_kpi_data.csv", parse_dates=["month"])
    return df

df = load_data()

st.title("📊 Business KPI Analytics Dashboard")
st.caption("Day 39 — Customer, Revenue & Retention KPI Monitoring")

# Sidebar controls
st.sidebar.header("Dashboard Controls")
min_date = df["month"].min().date()
max_date = df["month"].max().date()
date_range = st.sidebar.slider(
    "Analysis period",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)
view = df[(df["month"].dt.date >= date_range[0]) & (df["month"].dt.date <= date_range[1])].copy()

latest = view.iloc[-1]
previous = view.iloc[-2] if len(view) > 1 else view.iloc[-1]

def delta_pct(current, previous):
    if previous == 0:
        return 0
    return (current - previous) / abs(previous) * 100

# KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers", f'{latest["customers"]:,.0f}', f'{delta_pct(latest["customers"], previous["customers"]):+.1f}% MoM')
c2.metric("Revenue", f'${latest["revenue"]/1e6:.2f}M', f'{delta_pct(latest["revenue"], previous["revenue"]):+.1f}% MoM')
c3.metric("Retention", f'{latest["retention_rate"]*100:.1f}%', f'{(latest["retention_rate"]-previous["retention_rate"])*100:+.2f} pp')
c4.metric("ARPU", f'${latest["arpu"]:.2f}', f'{delta_pct(latest["arpu"], previous["arpu"]):+.1f}%')
c5.metric("AOV", f'${latest["avg_order_value"]:.2f}', f'{delta_pct(latest["avg_order_value"], previous["avg_order_value"]):+.1f}%')

st.divider()

left, right = st.columns(2)

with left:
    fig = px.line(view, x="month", y="revenue", markers=True,
                  title="Revenue Trend", labels={"revenue":"Revenue ($)", "month":"Month"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.line(view, x="month", y=["customers","active_customers"],
                  markers=True, title="Customer Base Trend",
                  labels={"value":"Customers", "month":"Month", "variable":"Metric"})
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    fig = px.line(view, x="month", y="retention_rate", markers=True,
                  title="Retention Rate Trend")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(view, x="month", y=["new_customers","churned_customers"],
                 barmode="group", title="Acquisition vs Churn",
                 labels={"value":"Customers","month":"Month","variable":"Metric"})
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Executive KPI Summary")
latest_month = latest["month"].strftime("%B %Y")
trend_revenue = "up" if latest["revenue_growth"] >= 0 else "down"
trend_customers = "up" if latest["customer_growth"] >= 0 else "down"
st.markdown(
    f"""
- **Revenue:** ${latest["revenue"]:,.0f} in {latest_month}, **{trend_revenue} {abs(latest["revenue_growth"])*100:.1f}%** month-over-month.
- **Customer base:** {latest["customers"]:,.0f} customers, **{trend_customers} {abs(latest["customer_growth"])*100:.1f}%** month-over-month.
- **Retention:** {latest["retention_rate"]*100:.1f}%; lower churn supports recurring revenue stability.
- **Unit economics:** ARPU is ${latest["arpu"]:.2f} and average order value is ${latest["avg_order_value"]:.2f}.
- **Management focus:** protect retention while scaling acquisition and revenue per customer.
"""
)

st.subheader("KPI Data")
st.dataframe(view, use_container_width=True)
