import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Customer Intelligence Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def demo_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n)
    tenure = rng.integers(1, 61, n)
    monthly = np.round(rng.uniform(20, 220, n), 2)
    support = rng.poisson(2.2, n)
    logins = np.maximum(1, rng.poisson(12, n))
    contract = rng.choice(["Monthly", "Annual", "Two Year"], n, p=[.52,.33,.15])
    segment = rng.choice(["Premium", "Standard", "Value"], n, p=[.25,.5,.25])
    churn_prob = 1/(1+np.exp(-(1.0*(contract=="Monthly")+0.09*support-0.035*tenure-0.035*logins-0.012*(monthly-100))))
    churn = rng.random(n) < np.clip(churn_prob, .03, .82)
    df = pd.DataFrame({
        "CustomerID":[f"C{i:05d}" for i in range(1,n+1)],
        "Age":age, "TenureMonths":tenure, "MonthlyRevenue":monthly,
        "SupportTickets":support, "MonthlyLogins":logins,
        "Contract":contract, "Segment":segment, "Churn":churn.astype(int),
        "ChurnRisk":np.round(churn_prob,3)
    })
    return df

st.title("📊 Customer Intelligence Dashboard")
st.caption("Interactive customer analytics, segmentation and churn-risk intelligence")

uploaded = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
df = pd.read_csv(uploaded) if uploaded else demo_data()

required = {"CustomerID","MonthlyRevenue","Segment","Churn"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

st.sidebar.header("Filters")
segments = st.sidebar.multiselect("Customer segment", sorted(df["Segment"].dropna().unique()), default=sorted(df["Segment"].dropna().unique()))
contracts = st.sidebar.multiselect("Contract", sorted(df["Contract"].dropna().unique()), default=sorted(df["Contract"].dropna().unique()) if "Contract" in df else [])
risk_cut = st.sidebar.slider("High-risk threshold", 0.0, 1.0, 0.60, 0.05)

filtered = df[df["Segment"].isin(segments)].copy()
if contracts and "Contract" in filtered:
    filtered = filtered[filtered["Contract"].isin(contracts)]

customers = len(filtered)
revenue = filtered["MonthlyRevenue"].sum()
churn_rate = filtered["Churn"].mean() if customers else 0
high_risk = (filtered["ChurnRisk"] >= risk_cut).sum() if "ChurnRisk" in filtered else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Customers", f"{customers:,}")
c2.metric("Monthly Revenue", f"${revenue:,.0f}")
c3.metric("Churn Rate", f"{churn_rate:.1%}")
c4.metric("High-Risk Customers", f"{high_risk:,}")

a,b = st.columns(2)
with a:
    seg = filtered.groupby("Segment", as_index=False).agg(Customers=("CustomerID","count"), Revenue=("MonthlyRevenue","sum"))
    fig = px.bar(seg, x="Segment", y="Customers", title="Customer Segmentation", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
with b:
    churn = filtered.groupby("Segment", as_index=False)["Churn"].mean()
    churn["ChurnRate"] = churn["Churn"]*100
    fig = px.bar(churn, x="Segment", y="ChurnRate", title="Churn Rate by Segment", text_auto=".1f")
    fig.update_yaxes(title="Churn rate (%)")
    st.plotly_chart(fig, use_container_width=True)

if "ChurnRisk" in filtered.columns:
    st.subheader("Churn Risk Distribution")
    fig = px.histogram(filtered, x="ChurnRisk", nbins=20, color="Churn", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("High-Risk Customer Watchlist")
cols = [c for c in ["CustomerID","Segment","Contract","MonthlyRevenue","TenureMonths","SupportTickets","MonthlyLogins","ChurnRisk","Churn"] if c in filtered.columns]
watch = filtered.sort_values("ChurnRisk", ascending=False)[cols].head(25)
st.dataframe(watch, use_container_width=True, hide_index=True)

st.info("Business interpretation: prioritize high-risk customers with low tenure, frequent support needs and weak engagement; use the dashboard as decision support rather than an automated decision-maker.")
