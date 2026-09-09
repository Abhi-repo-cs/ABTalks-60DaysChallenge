# day38.py - Customer Retention Analytics & CLV Estimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styles
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("crest")

def generate_synthetic_data(num_customers=1000):
    """Generates synthetic customer transaction history for retention analysis."""
    np.random.seed(42)
    customer_ids = [f"CUST_{i+1:04d}" for i in range(num_customers)]
    
    # Customer segments
    segment_weights = [0.15, 0.50, 0.35] # High-value, Mid-tier, At-risk/Low-value
    segments = np.random.choice(["High-Value", "Mid-Tier", "At-Risk"], size=num_customers, p=segment_weights)
    
    data = []
    for cid, seg in zip(customer_ids, segments):
        if seg == "High-Value":
            lifespan_months = np.random.randint(24, 48)
            monthly_rev = np.random.normal(250, 30)
            monthly_churn = 0.02
        elif seg == "Mid-Tier":
            lifespan_months = np.random.randint(12, 36)
            monthly_rev = np.random.normal(110, 20)
            monthly_churn = 0.05
        else: # At-Risk
            lifespan_months = np.random.randint(3, 18)
            monthly_rev = np.random.normal(45, 10)
            monthly_churn = 0.12
            
        data.append({
            "CustomerID": cid,
            "Segment": seg,
            "LifespanMonths": max(1, int(lifespan_months)),
            "AverageMonthlyRevenue": max(10.0, round(monthly_rev, 2)),
            "MonthlyChurnRate": monthly_churn
        })
    
    return pd.DataFrame(data)

def calculate_clv_metrics(df, discount_rate=0.10, margin=0.70):
    """
    Calculates Historical and Predictive CLV.
    
    Formula:
    - Historical CLV = Average Monthly Revenue * Lifespan Months * Margin
    - Predictive CLV = (Average Monthly Revenue * Margin) / (Monthly Churn Rate + Monthly Discount Rate)
    """
    monthly_discount = discount_rate / 12
    
    # Gross Margin Contribution per Month
    df["MonthlyMargin"] = df["AverageMonthlyRevenue"] * margin
    
    # Historical Realized CLV
    df["Historical_CLV"] = df["MonthlyMargin"] * df["LifespanMonths"]
    
    # Predictive CLV using Simplified Constant Churn Model
    df["Predictive_CLV"] = df["MonthlyMargin"] / (df["MonthlyChurnRate"] + monthly_discount)
    
    return df

def generate_visualizations(df):
    """Generates and saves visual reporting plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Predictive CLV Distribution by Segment
    sns.boxplot(
        data=df, 
        x="Segment", 
        y="Predictive_CLV", 
        ax=axes[0], 
        order=["High-Value", "Mid-Tier", "At-Risk"]
    )
    axes[0].set_title("Predictive CLV Distribution by Customer Segment", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Predicted Lifetime Value ($)")
    axes[0].set_xlabel("Customer Segment")
    
    # Plot 2: Monthly Revenue vs. Lifespan colored by Segment
    sns.scatterplot(
        data=df, 
        x="LifespanMonths", 
        y="AverageMonthlyRevenue", 
        hue="Segment", 
        ax=axes[1],
        alpha=0.8
    )
    axes[1].set_title("Customer Lifespan vs. Monthly Revenue", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lifespan (Months)")
    axes[1].set_ylabel("Avg Monthly Revenue ($)")
    
    plt.tight_layout()
    plt.savefig("clv_retention_report.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    df = generate_synthetic_data(1000)
    df = calculate_clv_metrics(df)
    
    # Summary Table
    summary = df.groupby("Segment").agg(
        Customer_Count=("CustomerID", "count"),
        Avg_Monthly_Revenue=("AverageMonthlyRevenue", "mean"),
        Avg_Lifespan_Months=("LifespanMonths", "mean"),
        Avg_Predictive_CLV=("Predictive_CLV", "mean")
    ).reset_index()
    
    print("--- CUSTOMER SEGMENTATION & CLV SUMMARY ---")
    print(summary.to_string(index=False))
    
    generate_visualizations(df)
    print("\n[+] Visualizations successfully rendered to 'clv_retention_report.png'")