# ============================================================
# DAY 31 - ADVANCED CUSTOMER INSIGHTS
# Customer Cluster -> Business Customer Personas
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "customer_clusters.csv"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("DAY 31 - ADVANCED CUSTOMER INSIGHTS")
print("=" * 60)

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"\nERROR: '{INPUT_FILE}' was not found.")
    print("Make sure your Day 30 clustered CSV is in the same folder.")
    raise SystemExit


print("\nDataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. AUTOMATICALLY FIND CLUSTER COLUMN
# ============================================================

possible_cluster_columns = [
    "Cluster",
    "cluster",
    "Clusters",
    "cluster_label",
    "Cluster_Label",
    "cluster_labels",
    "KMeans_Cluster"
]

cluster_column = None

for column in possible_cluster_columns:
    if column in df.columns:
        cluster_column = column
        break

if cluster_column is None:
    print("\nERROR: Cluster column not found.")
    print("Expected something like:")
    print("Cluster, cluster, cluster_label, etc.")
    raise SystemExit

print(f"\nCluster column detected: {cluster_column}")


# ============================================================
# 4. CLEAN CLUSTER COLUMN
# ============================================================

df = df.dropna(subset=[cluster_column]).copy()

print("\nNumber of clusters:",
      df[cluster_column].nunique())


# ============================================================
# 5. DETECT IMPORTANT FEATURES
# ============================================================

def find_column(possible_names):
    """
    Find a column using case-insensitive matching.
    """

    for name in possible_names:

        for column in df.columns:

            if column.lower().strip() == name.lower().strip():
                return column

    return None


income_column = find_column([
    "Annual Income",
    "Annual_Income",
    "Income",
    "Yearly Income",
    "Yearly_Income"
])

spending_column = find_column([
    "Spending Score",
    "Spending_Score",
    "SpendingScore",
    "Purchase Amount",
    "Purchase_Amount",
    "Total Spending",
    "Total_Spending"
])

frequency_column = find_column([
    "Purchase Frequency",
    "Purchase_Frequency",
    "Frequency",
    "PurchaseFrequency",
    "Number of Purchases",
    "Num Purchases"
])

recency_column = find_column([
    "Recency",
    "Days Since Last Purchase",
    "Days_Since_Last_Purchase",
    "Last Purchase Days"
])

age_column = find_column([
    "Age",
    "Customer Age",
    "Customer_Age"
])

engagement_column = find_column([
    "Engagement",
    "Engagement Score",
    "Engagement_Score"
])


print("\nDetected features:")

print("Income     :", income_column)
print("Spending   :", spending_column)
print("Frequency  :", frequency_column)
print("Recency    :", recency_column)
print("Age        :", age_column)
print("Engagement :", engagement_column)


# ============================================================
# 6. CONVERT NUMERIC FEATURES
# ============================================================

feature_columns = [
    income_column,
    spending_column,
    frequency_column,
    recency_column,
    age_column,
    engagement_column
]

feature_columns = [
    column for column in feature_columns
    if column is not None
]

for column in feature_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 7. CLUSTER PROFILE
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER PROFILE")
print("=" * 60)

cluster_profile = df.groupby(
    cluster_column
)[feature_columns].mean().round(2)

print("\nAverage characteristics by cluster:")
print(cluster_profile)


# Save cluster profile
cluster_profile.to_csv(
    f"{OUTPUT_DIR}/cluster_profile.csv"
)


# ============================================================
# 8. CUSTOMER COUNT BY CLUSTER
# ============================================================

cluster_counts = (
    df[cluster_column]
    .value_counts()
    .sort_index()
)

cluster_percentages = (
    cluster_counts / len(df) * 100
).round(2)

print("\nCustomer distribution:")

for cluster in cluster_counts.index:

    print(
        f"Cluster {cluster}: "
        f"{cluster_counts[cluster]} customers "
        f"({cluster_percentages[cluster]}%)"
    )


# ============================================================
# 9. NORMALIZED CLUSTER SCORES
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZED CUSTOMER SEGMENTATION")
print("=" * 60)

normalized_profile = cluster_profile.copy()

for column in normalized_profile.columns:

    minimum = normalized_profile[column].min()
    maximum = normalized_profile[column].max()

    if maximum != minimum:

        normalized_profile[column] = (
            normalized_profile[column] - minimum
        ) / (maximum - minimum)

    else:

        normalized_profile[column] = 0.5


# ============================================================
# 10. CALCULATE BUSINESS SCORES
# ============================================================

# Initialize scores

normalized_profile["Value Score"] = 0

normalized_profile["Engagement Score"] = 0

normalized_profile["Risk Score"] = 0


# Spending contribution
if spending_column is not None:

    normalized_profile["Value Score"] += (
        normalized_profile[spending_column] * 0.5
    )


# Income contribution
if income_column is not None:

    normalized_profile["Value Score"] += (
        normalized_profile[income_column] * 0.2
    )


# Frequency contribution
if frequency_column is not None:

    normalized_profile["Value Score"] += (
        normalized_profile[frequency_column] * 0.3
    )


# Engagement score

if frequency_column is not None:

    normalized_profile["Engagement Score"] += (
        normalized_profile[frequency_column] * 0.5
    )


if engagement_column is not None:

    normalized_profile["Engagement Score"] += (
        normalized_profile[engagement_column] * 0.5
    )


# Recency:
# Higher recency means the customer has been inactive longer.
# Therefore, higher normalized recency = higher risk.

if recency_column is not None:

    normalized_profile["Risk Score"] += (
        normalized_profile[recency_column] * 0.7
    )


# Low engagement increases risk

normalized_profile["Risk Score"] += (
    (1 - normalized_profile["Engagement Score"]) * 0.3
)


# ============================================================
# 11. AUTOMATIC PERSONA ASSIGNMENT
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER PERSONAS")
print("=" * 60)


def assign_persona(row):

    value = row["Value Score"]

    engagement = row["Engagement Score"]

    risk = row["Risk Score"]


    # High value + high engagement
    if value >= 0.65 and engagement >= 0.60:

        return "High-Value Loyalists"


    # High value + high risk
    elif value >= 0.60 and risk >= 0.55:

        return "At-Risk High-Value Customers"


    # High engagement but moderate value
    elif engagement >= 0.60:

        return "Engaged Growth Customers"


    # Low engagement + high risk
    elif risk >= 0.60:

        return "Low-Engagement Customers"


    # Moderate customers
    else:

        return "Potential Growth Customers"


normalized_profile["Persona"] = (
    normalized_profile.apply(
        assign_persona,
        axis=1
    )
)


for cluster, row in normalized_profile.iterrows():

    print(
        f"Cluster {cluster} → "
        f"{row['Persona']}"
    )


# ============================================================
# 12. MAP PERSONA BACK TO CUSTOMERS
# ============================================================

persona_mapping = normalized_profile[
    "Persona"
].to_dict()


df["Persona"] = df[
    cluster_column
].map(persona_mapping)


# ============================================================
# 13. PERSONA SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PERSONA SUMMARY")
print("=" * 60)


aggregation = {
    "Customer Count": (cluster_column, "count")
}


if income_column:
    aggregation["Average Income"] = (
        income_column,
        "mean"
    )


if spending_column:
    aggregation["Average Spending"] = (
        spending_column,
        "mean"
    )


if frequency_column:
    aggregation["Average Purchase Frequency"] = (
        frequency_column,
        "mean"
    )


if recency_column:
    aggregation["Average Recency"] = (
        recency_column,
        "mean"
    )


if age_column:
    aggregation["Average Age"] = (
        age_column,
        "mean"
    )


if engagement_column:
    aggregation["Average Engagement"] = (
        engagement_column,
        "mean"
    )


persona_summary = (
    df.groupby("Persona")
    .agg(**aggregation)
    .round(2)
)


# Add customer percentage

persona_summary["Customer Percentage"] = (
    persona_summary["Customer Count"]
    / len(df)
    * 100
).round(2)


print(persona_summary)


# Save summary

persona_summary.to_csv(
    f"{OUTPUT_DIR}/customer_persona_summary.csv"
)


# ============================================================
# 14. PERSONA CHARACTERISTICS
# ============================================================

print("\n" + "=" * 60)
print("PERSONA CHARACTERISTICS")
print("=" * 60)


for persona in persona_summary.index:

    print(f"\n### {persona}")

    persona_data = df[
        df["Persona"] == persona
    ]

    if spending_column:

        print(
            f"Average spending: "
            f"{persona_data[spending_column].mean():.2f}"
        )

    if frequency_column:

        print(
            f"Purchase frequency: "
            f"{persona_data[frequency_column].mean():.2f}"
        )

    if recency_column:

        print(
            f"Recency: "
            f"{persona_data[recency_column].mean():.2f}"
        )

    print(
        f"Customers: "
        f"{len(persona_data)}"
    )


# ============================================================
# 15. BUSINESS STRATEGY RECOMMENDATIONS
# ============================================================

strategies = {

    "High-Value Loyalists": {
        "Strategy": (
            "Focus on retention and loyalty."
        ),
        "Actions": (
            "VIP rewards, early product access, "
            "personalized recommendations, "
            "premium memberships and referral incentives."
        ),
        "Objective": (
            "Increase customer lifetime value."
        )
    },


    "At-Risk High-Value Customers": {
        "Strategy": (
            "Launch personalized win-back campaigns."
        ),
        "Actions": (
            "Exclusive offers, loyalty rewards, "
            "personalized communication, "
            "customer feedback and premium support."
        ),
        "Objective": (
            "Reduce churn and recover lost revenue."
        )
    },


    "Engaged Growth Customers": {
        "Strategy": (
            "Convert engagement into higher spending."
        ),
        "Actions": (
            "Cross-selling, upselling, product bundles, "
            "loyalty programs and personalized recommendations."
        ),
        "Objective": (
            "Increase customer value."
        )
    },


    "Low-Engagement Customers": {
        "Strategy": (
            "Reactivate inactive customers."
        ),
        "Actions": (
            "Targeted discounts, reminder campaigns, "
            "personalized offers and email/push campaigns."
        ),
        "Objective": (
            "Increase engagement and purchase frequency."
        )
    },


    "Potential Growth Customers": {
        "Strategy": (
            "Nurture customers toward higher-value behavior."
        ),
        "Actions": (
            "Product recommendations, bundles, "
            "loyalty rewards and targeted promotions."
        ),
        "Objective": (
            "Move customers into higher-value segments."
        )
    }
}


strategy_rows = []


for persona in persona_summary.index:

    strategy = strategies.get(
        persona,
        {
            "Strategy": "Develop a targeted customer strategy.",
            "Actions": "Use personalized marketing and engagement campaigns.",
            "Objective": "Increase customer value."
        }
    )

    strategy_rows.append({
        "Persona": persona,
        "Strategy": strategy["Strategy"],
        "Recommended Actions": strategy["Actions"],
        "Business Objective": strategy["Objective"]
    })


strategy_df = pd.DataFrame(
    strategy_rows
)


print("\nBusiness recommendations:")

print(strategy_df.to_string(index=False))


strategy_df.to_csv(
    f"{OUTPUT_DIR}/business_strategy_recommendations.csv",
    index=False
)


# ============================================================
# 16. VISUALIZATION 1
# CUSTOMER DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

persona_counts = (
    df["Persona"]
    .value_counts()
)

persona_counts.plot(
    kind="bar"
)

plt.title(
    "Customer Distribution by Persona"
)

plt.xlabel(
    "Customer Persona"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/customer_distribution_by_persona.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. VISUALIZATION 2
# AVERAGE SPENDING BY PERSONA
# ============================================================

if spending_column:

    spending_data = (
        df.groupby("Persona")[spending_column]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(10, 6))

    spending_data.plot(
        kind="bar"
    )

    plt.title(
        "Average Spending by Customer Persona"
    )

    plt.xlabel(
        "Customer Persona"
    )

    plt.ylabel(
        "Average Spending"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/average_spending_by_persona.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 18. VISUALIZATION 3
# SPENDING VS FREQUENCY
# ============================================================

if spending_column and frequency_column:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x=frequency_column,
        y=spending_column,
        hue="Persona",
        s=80
    )

    plt.title(
        "Spending vs Purchase Frequency"
    )

    plt.xlabel(
        "Purchase Frequency"
    )

    plt.ylabel(
        "Spending"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/spending_vs_frequency.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 19. VISUALIZATION 4
# INCOME VS SPENDING
# ============================================================

if income_column and spending_column:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x=income_column,
        y=spending_column,
        hue="Persona",
        s=80
    )

    plt.title(
        "Income vs Spending by Customer Persona"
    )

    plt.xlabel(
        "Annual Income"
    )

    plt.ylabel(
        "Spending"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/income_vs_spending.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 20. VISUALIZATION 5
# RECENCY BY PERSONA
# ============================================================

if recency_column:

    recency_data = (
        df.groupby("Persona")[recency_column]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(10, 6))

    recency_data.plot(
        kind="bar"
    )

    plt.title(
        "Average Recency by Customer Persona"
    )

    plt.xlabel(
        "Customer Persona"
    )

    plt.ylabel(
        "Average Recency"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/recency_by_persona.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 21. HEATMAP OF PERSONA CHARACTERISTICS
# ============================================================

heatmap_columns = [
    column
    for column in feature_columns
    if column in normalized_profile.columns
]


if len(heatmap_columns) >= 2:

    plt.figure(
        figsize=(12, 7)
    )

    sns.heatmap(
        normalized_profile[heatmap_columns],
        annot=True,
        fmt=".2f",
        cmap="viridis"
    )

    plt.title(
        "Normalized Customer Persona Characteristics"
    )

    plt.xlabel(
        "Customer Features"
    )

    plt.ylabel(
        "Customer Cluster"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/persona_heatmap.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 22. CREATE FINAL PERSONA DATASET
# ============================================================

df.to_csv(
    f"{OUTPUT_DIR}/customer_personas.csv",
    index=False
)


# ============================================================
# 23. GENERATE MARKDOWN REPORT
# ============================================================

report_path = (
    f"{OUTPUT_DIR}/customer_persona_report.md"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "# Advanced Customer Insights – Day 31\n\n"
    )

    report.write(
        "## Objective\n\n"
    )

    report.write(
        "Transform machine-learning customer clusters "
        "into meaningful business personas and develop "
        "strategies for each customer segment.\n\n"
    )


    report.write(
        "## Dataset Overview\n\n"
    )

    report.write(
        f"- Total customers: {len(df)}\n"
    )

    report.write(
        f"- Number of clusters: "
        f"{df[cluster_column].nunique()}\n\n"
    )


    report.write(
        "## Customer Personas\n\n"
    )


    for persona in persona_summary.index:

        report.write(
            f"### {persona}\n\n"
        )

        count = persona_summary.loc[
            persona,
            "Customer Count"
        ]

        percentage = persona_summary.loc[
            persona,
            "Customer Percentage"
        ]

        report.write(
            f"- Customers: {count}\n"
        )

        report.write(
            f"- Customer share: {percentage}%\n"
        )


        if spending_column:

            value = persona_summary.loc[
                persona,
                "Average Spending"
            ]

            report.write(
                f"- Average spending: {value:.2f}\n"
            )


        if frequency_column:

            value = persona_summary.loc[
                persona,
                "Average Purchase Frequency"
            ]

            report.write(
                f"- Average purchase frequency: {value:.2f}\n"
            )


        if recency_column:

            value = persona_summary.loc[
                persona,
                "Average Recency"
            ]

            report.write(
                f"- Average recency: {value:.2f}\n"
            )


        strategy = strategies.get(
            persona,
            {}
        )


        report.write(
            f"- **Strategy:** "
            f"{strategy.get('Strategy', 'Targeted marketing')}\n"
        )

        report.write(
            f"- **Actions:** "
            f"{strategy.get('Actions', 'Personalized engagement')}\n"
        )

        report.write(
            f"- **Business Objective:** "
            f"{strategy.get('Objective', 'Increase customer value')}\n\n"
        )


    report.write(
        "## Business Recommendations\n\n"
    )


    for _, row in strategy_df.iterrows():

        report.write(
            f"### {row['Persona']}\n\n"
        )

        report.write(
            f"**Strategy:** {row['Strategy']}\n\n"
        )

        report.write(
            f"**Recommended Actions:** "
            f"{row['Recommended Actions']}\n\n"
        )

        report.write(
            f"**Business Objective:** "
            f"{row['Business Objective']}\n\n"
        )


    report.write(
        "## Conclusion\n\n"
    )

    report.write(
        "The clustering analysis was transformed into "
        "actionable customer personas. Each persona "
        "represents a distinct behavioral pattern and "
        "can be targeted with a different business "
        "strategy. This approach demonstrates how "
        "unsupervised machine learning can support "
        "customer retention, personalization, "
        "cross-selling and customer lifetime value optimization.\n"
    )


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("DAY 31 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

for file in os.listdir(OUTPUT_DIR):

    print(
        f"✓ {file}"
    )


print("\nPersona mapping:")

for cluster, persona in persona_mapping.items():

    print(
        f"Cluster {cluster} → {persona}"
    )

print("\nAll results have been saved inside:")
print(f"'{OUTPUT_DIR}/'")