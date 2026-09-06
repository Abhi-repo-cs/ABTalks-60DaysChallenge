import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from data import load_data


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Day 35 | Unified Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Day 35 — Unified Analytics Platform")

st.caption(
    "Sprint Review & System Integration | "
    "Customer Segmentation + Recommendations + Anomaly Detection"
)


# ============================================================
# LOAD DATA
# ============================================================

df, data_source = load_data()

if df.empty:
    st.error("No data available.")
    st.stop()


# ============================================================
# DATA PREPROCESSING
# ============================================================

df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
).fillna(0)

df["freight_value"] = pd.to_numeric(
    df["freight_value"],
    errors="coerce"
).fillna(0)

df = df.dropna(
    subset=["customer_id", "order_id"]
)


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

@st.cache_data
def create_segments(data):

    reference_date = (
        data["order_date"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        data.groupby("customer_id")
        .agg(
            recency=(
                "order_date",
                lambda x:
                (reference_date - x.max()).days
            ),

            frequency=(
                "order_id",
                "nunique"
            ),

            monetary=(
                "price",
                "sum"
            )
        )
        .reset_index()
    )

    rfm = rfm.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    features = rfm[
        [
            "recency",
            "frequency",
            "monetary"
        ]
    ].copy()

    # Reduce monetary skew
    features["monetary"] = np.log1p(
        features["monetary"]
    )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    n_clusters = min(
        5,
        max(
            2,
            len(rfm) // 100
        )
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = model.fit_predict(
        scaled_features
    )

    # --------------------------------------------------------
    # Create business-friendly labels
    # --------------------------------------------------------

    summary = (
        rfm.groupby("cluster")
        .agg(
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean")
        )
        .reset_index()
    )

    summary["score"] = (
        summary["avg_frequency"].rank()
        +
        summary["avg_monetary"].rank()
        -
        summary["avg_recency"].rank()
    )

    ordered = summary.sort_values(
        "score",
        ascending=False
    )

    labels = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "At Risk",
        "Low Value"
    ]

    mapping = {}

    for i, cluster in enumerate(
        ordered["cluster"]
    ):

        if i < len(labels):

            mapping[
                cluster
            ] = labels[i]

        else:

            mapping[
                cluster
            ] = f"Segment {i + 1}"

    rfm["segment"] = (
        rfm["cluster"]
        .map(mapping)
    )

    return rfm


rfm = create_segments(df)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

@st.cache_data
def create_recommendations(data):

    # Most purchased products
    popularity = (
        data.groupby("product_id")
        .agg(
            purchases=(
                "order_id",
                "nunique"
            ),

            revenue=(
                "price",
                "sum"
            )
        )
        .reset_index()
        .sort_values(
            "purchases",
            ascending=False
        )
    )

    # Products already purchased by each customer
    purchased = (
        data.groupby("customer_id")[
            "product_id"
        ]
        .apply(set)
        .to_dict()
    )

    popular_products = (
        popularity["product_id"]
        .tolist()
    )

    recommendations = {}

    for customer, products in purchased.items():

        recs = [
            product
            for product in popular_products
            if product not in products
        ]

        recommendations[
            customer
        ] = recs[:5]

    return recommendations, popularity


recommendations, popularity = (
    create_recommendations(df)
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

@st.cache_data
def detect_anomalies(data):

    transactions = (
        data.groupby("order_id")
        .agg(

            order_value=(
                "price",
                "sum"
            ),

            freight=(
                "freight_value",
                "sum"
            ),

            items=(
                "product_id",
                "count"
            )
        )
        .reset_index()
    )

    transactions = transactions.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(transactions) < 20:

        transactions[
            "anomaly"
        ] = 1

        return transactions

    features = transactions[
        [
            "order_value",
            "freight",
            "items"
        ]
    ]

    model = IsolationForest(
        contamination=0.03,
        random_state=42
    )

    transactions[
        "anomaly"
    ] = model.fit_predict(
        features
    )

    return transactions


anomalies = detect_anomalies(df)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def business_insight(segment):

    insights = {

        "Champions":
        "Focus on loyalty programs, premium offers, "
        "and personalized recommendations.",

        "Loyal Customers":
        "Use cross-selling and complementary product "
        "recommendations to increase purchase value.",

        "Potential Loyalists":
        "Use targeted promotions to encourage repeat "
        "purchases.",

        "At Risk":
        "Launch re-engagement campaigns and recommend "
        "previously relevant products.",

        "Low Value":
        "Use cost-efficient marketing strategies and "
        "monitor customer acquisition efficiency."
    }

    return insights.get(
        segment,
        "Monitor this customer group."
    )


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = (
    df["customer_id"]
    .nunique()
)

total_orders = (
    df["order_id"]
    .nunique()
)

total_revenue = (
    df["price"]
    .sum()
)

average_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)

anomaly_count = int(
    (
        anomalies["anomaly"] == -1
    ).sum()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Overview",
        "Customer Segmentation",
        "Recommendations",
        "Anomaly Detection",
        "Business Insights",
        "System Architecture"
    ]
)

st.sidebar.info(
    f"Data Source: {data_source}"
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Revenue",
        f"₹{total_revenue:,.0f}"
    )

    col4.metric(
        "Avg Order Value",
        f"₹{average_order_value:,.2f}"
    )

    col5.metric(
        "Anomalies",
        f"{anomaly_count:,}"
    )

    st.divider()

    left, right = st.columns(2)

    # --------------------------------------------------------
    # SEGMENTS
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Customer Segments"
        )

        segment_counts = (
            rfm["segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "segment",
            "customers"
        ]

        fig = px.bar(
            segment_counts,
            x="segment",
            y="customers",
            title="Customer Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # REVENUE BY SEGMENT
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Revenue by Segment"
        )

        customer_revenue = (
            df.groupby(
                "customer_id"
            )["price"]
            .sum()
            .rename("revenue")
        )

        revenue_segment = (
            rfm.merge(
                customer_revenue,
                on="customer_id"
            )
            .groupby("segment")[
                "revenue"
            ]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            revenue_segment,
            names="segment",
            values="revenue",
            title="Revenue Contribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # REVENUE TREND
    # --------------------------------------------------------

    st.subheader(
        "Monthly Revenue Trend"
    )

    trend = (
        df.dropna(
            subset=["order_date"]
        )
        .set_index("order_date")
        .resample("ME")["price"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="order_date",
        y="price",
        markers=True,
        title="Monthly Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "Customer Segmentation":

    st.header(
        "👥 Customer Segmentation"
    )

    st.write(
        "Customers are grouped using RFM "
        "features and K-Means clustering."
    )

    st.subheader(
        "Customer RFM Table"
    )

    st.dataframe(
        rfm[
            [
                "customer_id",
                "recency",
                "frequency",
                "monetary",
                "segment"
            ]
        ].head(100),
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        fig = px.scatter(
            rfm,
            x="frequency",
            y="monetary",
            color="segment",
            hover_data=[
                "customer_id",
                "recency"
            ],
            title="Customer Behavioral Segments"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        summary = (
            rfm.groupby("segment")
            .agg(
                customers=(
                    "customer_id",
                    "count"
                ),

                avg_frequency=(
                    "frequency",
                    "mean"
                ),

                avg_monetary=(
                    "monetary",
                    "mean"
                ),

                avg_recency=(
                    "recency",
                    "mean"
                )
            )
            .reset_index()
        )

        st.dataframe(
            summary,
            use_container_width=True
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "Recommendations":

    st.header(
        "🛍️ Product Recommendation Engine"
    )

    customer = st.selectbox(
        "Select Customer",
        sorted(
            recommendations.keys()
        )
    )

    customer_data = rfm[
        rfm["customer_id"] == customer
    ]

    if not customer_data.empty:

        segment = (
            customer_data.iloc[0][
                "segment"
            ]
        )

        st.metric(
            "Customer Segment",
            segment
        )

        st.info(
            business_insight(segment)
        )

    st.subheader(
        "Recommended Products"
    )

    recs = recommendations.get(
        customer,
        []
    )

    if recs:

        for i, product in enumerate(
            recs,
            1
        ):

            st.write(
                f"### {i}. {product}"
            )

    else:

        st.warning(
            "No recommendations available."
        )

    st.subheader(
        "Most Popular Products"
    )

    fig = px.bar(
        popularity.head(20),
        x="product_id",
        y="purchases",
        title="Top 20 Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.header(
        "🚨 Anomaly Detection"
    )

    normal_count = int(
        (
            anomalies["anomaly"] == 1
        ).sum()
    )

    anomaly_count = int(
        (
            anomalies["anomaly"] == -1
        ).sum()
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Transactions",
        f"{len(anomalies):,}"
    )

    col2.metric(
        "Normal",
        f"{normal_count:,}"
    )

    col3.metric(
        "Anomalies",
        f"{anomaly_count:,}"
    )

    plot_data = anomalies.copy()

    plot_data["Status"] = (
        plot_data["anomaly"]
        .map({
            1: "Normal",
            -1: "Anomaly"
        })
    )

    fig = px.scatter(
        plot_data,
        x="order_value",
        y="freight",
        size="items",
        color="Status",
        hover_data=[
            "order_id"
        ],
        title="Transaction Anomaly Detection"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Detected Anomalies"
    )

    st.dataframe(
        anomalies[
            anomalies["anomaly"] == -1
        ]
        .sort_values(
            "order_value",
            ascending=False
        ),
        use_container_width=True
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.header(
        "💡 Business Decision Support"
    )

    summary = (
        rfm.groupby("segment")
        .agg(
            customers=(
                "customer_id",
                "count"
            ),

            avg_frequency=(
                "frequency",
                "mean"
            ),

            avg_spend=(
                "monetary",
                "mean"
            ),

            avg_recency=(
                "recency",
                "mean"
            )
        )
        .reset_index()
    )

    for _, row in summary.iterrows():

        segment = row["segment"]

        with st.expander(
            f"{segment} — "
            f"{int(row['customers']):,} customers"
        ):

            st.write(
                f"Average Frequency: "
                f"{row['avg_frequency']:.2f}"
            )

            st.write(
                f"Average Monetary Value: "
                f"{row['avg_spend']:.2f}"
            )

            st.write(
                f"Average Recency: "
                f"{row['avg_recency']:.1f} days"
            )

            st.success(
                business_insight(segment)
            )

    st.divider()

    st.subheader(
        "Recommended Business Actions"
    )

    actions = [
        "Prioritize Champions with loyalty programs.",
        "Use recommendations to increase cross-selling.",
        "Re-engage At-Risk customers with targeted campaigns.",
        "Investigate anomalous transactions.",
        "Use customer segments to optimize marketing spend."
    ]

    for action in actions:

        st.write(
            f"✅ {action}"
        )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

elif page == "System Architecture":

    st.header(
        "🏗️ Workflow Architecture"
    )

    st.code(
        """
                 RAW DATA
                     |
                     v
           DATA PREPROCESSING
                     |
                     v
          FEATURE ENGINEERING
                     |
        +------------+------------+
        |            |            |
        v            v            v
   SEGMENTATION  RECOMMENDATION  ANOMALY
        |            |            |
        +------------+------------+
                     |
                     v
            UNIFIED ANALYTICS
                     |
                     v
           INTERACTIVE DASHBOARD
                     |
                     v
             BUSINESS ACTIONS
        """,
        language="text"
    )

    st.subheader(
        "System Components"
    )

    components = pd.DataFrame({
        "Component": [
            "Data Layer",
            "Segmentation",
            "Recommendation",
            "Anomaly Detection",
            "Dashboard",
            "Decision Layer"
        ],

        "Technology": [
            "Pandas / CSV",
            "RFM + K-Means",
            "Purchase History",
            "Isolation Forest",
            "Streamlit + Plotly",
            "Business Rules"
        ]
    })

    st.dataframe(
        components,
        use_container_width=True
    )

    st.success(
        "The system integrates multiple Week 5 "
        "analytics modules into a single "
        "decision-support workflow."
    )