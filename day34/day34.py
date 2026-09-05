
import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    color: #666;
    font-size: 16px;
    margin-bottom: 25px;
}

.kpi-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f7f9fc;
    border: 1px solid #e5e7eb;
    text-align: center;
}

.kpi-title {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
}

.insight-box {
    padding: 18px;
    border-radius: 10px;
    background-color: #f7f9fc;
    border-left: 5px solid #4f46e5;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("customer_data.csv")

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Convert date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    # Convert revenue to numeric
    if "monthly_revenue" in df.columns:
        df["monthly_revenue"] = pd.to_numeric(
            df["monthly_revenue"],
            errors="coerce"
        ).fillna(0)

    return df


# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = load_data()

except FileNotFoundError:

    st.error(
        "customer_data.csv was not found. "
        "Place customer_data.csv in the same folder as day34.py."
    )

    st.stop()


# =========================================================
# DATA VALIDATION
# =========================================================

required_columns = [
    "customer_id",
    "date",
    "segment",
    "region",
    "plan",
    "monthly_revenue",
    "churn",
    "churn_risk"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Missing columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# =========================================================
# CREATE CHURN FLAG
# =========================================================

df["churn_flag"] = (
    df["churn"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "yes",
            "true",
            "1",
            "churned"
        ]
    )
    .astype(int)
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '📊 Customer Analytics & Churn Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Executive Business Intelligence Dashboard | Day 34'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎛️ Dashboard Filters")

st.sidebar.markdown(
    "Use the filters below to explore customer behavior."
)


# Segment filter

segment_options = sorted(
    df["segment"]
    .dropna()
    .unique()
)

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=segment_options,
    default=segment_options
)


# Region filter

region_options = sorted(
    df["region"]
    .dropna()
    .unique()
)

selected_regions = st.sidebar.multiselect(
    "Region",
    options=region_options,
    default=region_options
)


# Plan filter

plan_options = sorted(
    df["plan"]
    .dropna()
    .unique()
)

selected_plans = st.sidebar.multiselect(
    "Subscription Plan",
    options=plan_options,
    default=plan_options
)


# Churn filter

churn_options = sorted(
    df["churn"]
    .dropna()
    .unique()
)

selected_churn = st.sidebar.multiselect(
    "Churn Status",
    options=churn_options,
    default=churn_options
)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df[
    df["segment"].isin(selected_segments)
    & df["region"].isin(selected_regions)
    & df["plan"].isin(selected_plans)
    & df["churn"].isin(selected_churn)
].copy()


# =========================================================
# HANDLE EMPTY DATA
# =========================================================

if filtered_df.empty:

    st.warning(
        "No customers match the selected filters. "
        "Please adjust the filters."
    )

    st.stop()


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_customers = len(filtered_df)

churned_customers = filtered_df[
    filtered_df["churn_flag"] == 1
].shape[0]

churn_rate = (
    churned_customers /
    total_customers *
    100
)

total_monthly_revenue = (
    filtered_df["monthly_revenue"].sum()
)

revenue_at_risk = (
    filtered_df.loc[
        filtered_df["churn_flag"] == 1,
        "monthly_revenue"
    ].sum()
)

high_risk_customers = filtered_df[
    filtered_df["churn_risk"]
    .astype(str)
    .str.lower()
    == "high"
].shape[0]


# =========================================================
# KPI CARDS
# =========================================================

st.subheader("Executive Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Churn Rate</div>
            <div class="kpi-value">{churn_rate:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Revenue at Risk</div>
            <div class="kpi-value">${revenue_at_risk:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">High-Risk Customers</div>
            <div class="kpi-value">{high_risk_customers:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================

st.subheader("Customer Segmentation")

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Segment Distribution
# ---------------------------------------------------------

with col1:

    segment_counts = (
        filtered_df["segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "segment",
        "customers"
    ]

    fig_segment = px.pie(
        segment_counts,
        names="segment",
        values="customers",
        hole=0.45,
        title="Customer Distribution by Segment"
    )

    fig_segment.update_layout(
        legend_title="Segment"
    )

    st.plotly_chart(
        fig_segment,
        use_container_width=True
    )


# ---------------------------------------------------------
# Revenue by Segment
# ---------------------------------------------------------

with col2:

    revenue_segment = (
        filtered_df
        .groupby("segment")["monthly_revenue"]
        .sum()
        .reset_index()
    )

    fig_revenue = px.bar(
        revenue_segment,
        x="segment",
        y="monthly_revenue",
        title="Monthly Revenue by Segment",
        labels={
            "segment": "Customer Segment",
            "monthly_revenue": "Monthly Revenue ($)"
        }
    )

    st.plotly_chart(
        fig_revenue,
        use_container_width=True
    )


# =========================================================
# CHURN ANALYSIS
# =========================================================

st.subheader("Churn Analysis")

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Churn by Segment
# ---------------------------------------------------------

with col1:

    churn_segment = (
        filtered_df
        .groupby("segment")["churn_flag"]
        .mean()
        .reset_index()
    )

    # IMPORTANT:
    # Rename churn_flag to churn_rate
    # before using churn_rate.

    churn_segment = churn_segment.rename(
        columns={
            "churn_flag": "churn_rate"
        }
    )

    # Convert decimal to percentage

    churn_segment["churn_rate"] *= 100

    fig_churn = px.bar(
        churn_segment,
        x="segment",
        y="churn_rate",
        title="Churn Rate by Customer Segment",
        labels={
            "segment": "Customer Segment",
            "churn_rate": "Churn Rate (%)"
        },
        text="churn_rate"
    )

    fig_churn.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_churn.update_yaxes(
        ticksuffix="%"
    )

    st.plotly_chart(
        fig_churn,
        use_container_width=True
    )


# ---------------------------------------------------------
# Churn by Plan
# ---------------------------------------------------------

with col2:

    churn_plan = (
        filtered_df
        .groupby("plan")["churn_flag"]
        .mean()
        .reset_index()
    )

    churn_plan = churn_plan.rename(
        columns={
            "churn_flag": "churn_rate"
        }
    )

    churn_plan["churn_rate"] *= 100

    fig_plan = px.bar(
        churn_plan,
        x="plan",
        y="churn_rate",
        title="Churn Rate by Subscription Plan",
        labels={
            "plan": "Subscription Plan",
            "churn_rate": "Churn Rate (%)"
        },
        text="churn_rate"
    )

    fig_plan.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_plan.update_yaxes(
        ticksuffix="%"
    )

    st.plotly_chart(
        fig_plan,
        use_container_width=True
    )


# =========================================================
# CHURN RISK DISTRIBUTION
# =========================================================

st.subheader("Customer Risk Distribution")

risk_counts = (
    filtered_df["churn_risk"]
    .value_counts()
    .reset_index()
)

risk_counts.columns = [
    "risk_level",
    "customers"
]

fig_risk = px.bar(
    risk_counts,
    x="risk_level",
    y="customers",
    title="Customers by Churn Risk Level",
    labels={
        "risk_level": "Risk Level",
        "customers": "Number of Customers"
    },
    text="customers"
)

fig_risk.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# =========================================================
# TREND ANALYSIS
# =========================================================

st.subheader("Customer Trends Over Time")

trend_df = filtered_df.copy()

trend_df["month"] = (
    trend_df["date"]
    .dt.to_period("M")
    .astype(str)
)


# ---------------------------------------------------------
# Monthly Customer Count
# ---------------------------------------------------------

monthly_customers = (
    trend_df
    .groupby("month")
    .size()
    .reset_index(
        name="customers"
    )
)

fig_trend = px.line(
    monthly_customers,
    x="month",
    y="customers",
    markers=True,
    title="Monthly Customer Volume",
    labels={
        "month": "Month",
        "customers": "Customers"
    }
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)


# =========================================================
# MONTHLY CHURN TREND
# =========================================================

monthly_churn = (
    trend_df
    .groupby("month")["churn_flag"]
    .mean()
    .reset_index()
)

monthly_churn["churn_rate"] = (
    monthly_churn["churn_flag"] * 100
)

fig_monthly_churn = px.line(
    monthly_churn,
    x="month",
    y="churn_rate",
    markers=True,
    title="Monthly Churn Rate",
    labels={
        "month": "Month",
        "churn_rate": "Churn Rate (%)"
    }
)

fig_monthly_churn.update_yaxes(
    ticksuffix="%"
)

st.plotly_chart(
    fig_monthly_churn,
    use_container_width=True
)


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.divider()

st.subheader("💡 Business Insights")


# ---------------------------------------------------------
# Highest Churn Segment
# ---------------------------------------------------------

segment_churn = (
    filtered_df
    .groupby("segment")["churn_flag"]
    .mean()
    .sort_values(
        ascending=False
    )
)

highest_churn_segment = (
    segment_churn.index[0]
)

highest_churn_rate = (
    segment_churn.iloc[0] * 100
)


# ---------------------------------------------------------
# Largest Revenue Segment
# ---------------------------------------------------------

segment_revenue = (
    filtered_df
    .groupby("segment")["monthly_revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
)

largest_revenue_segment = (
    segment_revenue.index[0]
)


# ---------------------------------------------------------
# Insights
# ---------------------------------------------------------

st.markdown(
    f"""
    <div class="insight-box">
        <strong>🔎 Highest Churn Segment</strong><br>
        {highest_churn_segment} has the highest churn rate
        at <strong>{highest_churn_rate:.1f}%</strong>.
        This segment should receive additional retention attention.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="insight-box">
        <strong>💰 Revenue Exposure</strong><br>
        Approximately
        <strong>${revenue_at_risk:,.0f}</strong>
        in monthly revenue is associated with churned customers.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="insight-box">
        <strong>📈 Revenue Contribution</strong><br>
        The <strong>{largest_revenue_segment}</strong> segment
        currently contributes the largest amount of monthly revenue.
    </div>
    """,
    unsafe_allow_html=True
)


if churn_rate >= 20:

    st.warning(
        f"⚠️ Overall churn is {churn_rate:.1f}%. "
        "Retention should be treated as a high-priority business issue."
    )

elif churn_rate >= 10:

    st.info(
        f"ℹ️ Overall churn is {churn_rate:.1f}%. "
        "Customer retention should be actively monitored."
    )

else:

    st.success(
        f"✅ Overall churn is {churn_rate:.1f}%. "
        "Current customer retention appears relatively stable."
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

st.subheader("🎯 Recommended Business Actions")

recommendations = [
    "Prioritize retention campaigns for high-risk customers.",
    "Investigate why the highest-churn segment has elevated attrition.",
    "Monitor revenue at risk rather than focusing only on customer count.",
    "Use customer segmentation to personalize retention strategies.",
    "Track churn trends monthly to identify early deterioration."
]

for recommendation in recommendations:

    st.write(
        f"• {recommendation}"
    )


# =========================================================
# CUSTOMER DATA
# =========================================================

st.divider()

st.subheader("📋 Filtered Customer Data")

st.caption(
    f"Showing {len(filtered_df):,} customers based on the selected filters."
)

display_columns = [
    "customer_id",
    "date",
    "segment",
    "region",
    "plan",
    "monthly_revenue",
    "churn",
    "churn_risk"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Day 34 | Business Intelligence Dashboards | "
    "Customer Analytics & Churn Intelligence"
)

