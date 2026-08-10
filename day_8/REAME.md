# Sales data exploratory data analysis (EDA)

## Overview

This project performs exploratory data analysis (EDA) on a cleaned sales dataset using **Pandas, Matplotlib, and Seaborn**. The objective is to understand sales performance, identify trends, detect anomalies, and generate business insights from the data.

## Dataset

The dataset contains **1,000 sales records** with information such as:

- Sale date
- Sales representative
- Region
- Sales amount
- Quantity sold
- Product category
- Unit cost
- Unit price
- Customer type
- Discount
- Payment method
- Sales channel
- Total sales
- Sale month

## EDA tasks performed

### Summary statistics

- Examined data types and missing values
- Generated descriptive statistics using `df.describe()`
- Reviewed distributions of numeric and categorical features

### Visualizations

The following visualizations were created:

- Distribution of total sales (Histogram + KDE)
- Box plot of total sales
- Quantity sold vs total sales (Scatter plot)
- Correlation matrix (Heatmap)
- Total sales by region (Bar chart)
- Monthly sales trend (Line chart)

## Key findings

### 1. Sales distribution is right-skewed

The histogram shows that most transactions fall within the **lower sales range**, while a smaller number of transactions generate very high revenue.

**Business implication:** A few high-value orders contribute significantly to total revenue.

### 2. High-value sales outliers exist

The box plot reveals several transactions above **230,000–250,000**, indicating unusually large sales orders.

**Business implication:** These may represent enterprise customers, bulk purchases, or exceptional sales events that should be analyzed separately.

### 3. Quantity sold strongly influences revenue

The scatter plot shows a clear positive relationship between **Quantity Sold** and **Total Sales**.

**Business implication:** Increasing average order quantity can directly improve revenue.

### 4. Quantity sold has the strongest correlation with total sales

From the correlation matrix:

- **Quantity Sold ↔ Total Sales: 0.71**
- **Unit Price ↔ Total Sales: 0.66**
- **Unit Cost ↔ Total Sales: 0.65**
- **Discount ↔ Total Sales: -0.015**

**Business implication:** Revenue is primarily driven by sales volume rather than discounting.

### 5. Regional sales performance is relatively balanced

The regional sales chart indicates:

- **North** has the highest total sales
- **East** and **West** follow closely
- **South** has the lowest sales

**Business implication:** The South region may require targeted marketing, sales expansion, or product optimization.

### 6. Monthly sales fluctuate significantly

The monthly trend shows:

- A peak in **January**
- Another strong period around **October**
- Lower sales during **February** and **November**

**Business implication:** Sales are seasonal, and inventory and promotional planning should account for these fluctuations.

## Business recommendations

- Focus on increasing **average order quantity**
- Investigate **high-value outlier transactions**
- Strengthen sales efforts in the **South region**
- Prepare inventory for **January and October demand peaks**
- Optimize pricing and product mix instead of relying heavily on discounts

## Technologies used

- Python
- Pandas
- Matplotlib
- Seaborn

## Project structure

```text
sales-eda/
│── sales_data_cleaned.csv
│── day8.py
│── plots/
│   ├── histogram_total_sales.png
│   ├── boxplot_total_sales.png
│   ├── quantity_vs_sales.png
│   ├── correlation_matrix.png
│   ├── regional_sales.png
│   └── monthly_sales_trend.png
└── README.md
```

## Conclusion

The EDA reveals that **sales volume is the primary driver of revenue**, while regional performance and seasonal demand create meaningful business opportunities. The analysis provides a strong foundation for future forecasting, customer segmentation, and sales optimization models.