# Day 7 – Exploratory Data Analysis (EDA)

## Dataset: Sales Data

This analysis explores the relationship between **Total_Sales** and **Quantity_Sold** using a scatter plot and examines the distribution of **Total_Sales** using a histogram.

---

# Scatter Plot Analysis (Total_Sales vs Quantity_Sold)

## Patterns observed

- A clear **positive relationship** exists between **Total_Sales** and **Quantity_Sold**.
- As **Total_Sales** increases, **Quantity_Sold** generally increases.
- The points form a **fan-shaped pattern**, indicating increasing variability at higher sales values.
- Most transactions are concentrated in the **lower sales range (below 100,000)**.
- The relationship is strong but not perfectly linear, which is consistent with the correlation value of **0.705**.

## Outliers observed

- A few transactions have **very high Total_Sales (₹200,000–₹250,000)** with quantities close to **45–50 units**.
- Some points show **high sales with relatively low quantities**, suggesting premium-priced products.
- A few observations have **high quantities but only moderate sales**, indicating lower-priced products.

## Five insights

1. **Quantity_Sold has a strong positive relationship with Total_Sales (correlation = 0.705).**
2. **Higher quantities generally lead to higher revenue generation.**
3. **The spread of points increases with Total_Sales, suggesting other factors such as price, discounts, and product category also influence sales.**
4. **High-value transactions are relatively rare compared with lower-value transactions.**
5. **The presence of both high-price and low-price products creates variation in Total_Sales for similar quantities sold.**

---

# Histogram Analysis (Distribution of Total_Sales)

## Patterns observed

- The distribution is **positively skewed (right-skewed)**.
- Most transactions fall between **₹0 and 50,000**.
- Frequency decreases steadily as sales values increase.
- The histogram has a **long right tail**, indicating a small number of very large transactions.
- The data is **not normally distributed**.

## Outliers observed

- Transactions above **₹200,000** appear infrequently and can be considered **high-value outliers**.
- These extreme values are much larger than the majority of sales transactions.

## Five insights

1. **Most sales transactions are relatively small, with the highest concentration below 50,000.**
2. **The distribution is heavily right-skewed, meaning a few transactions generate exceptionally high revenue.**
3. **High-value sales are uncommon but may contribute significantly to total revenue.**
4. **The sales distribution is uneven, indicating revenue concentration among a small number of large transactions.**
5. **Because the data is skewed, techniques such as log transformation may be useful for statistical modeling and machine learning.**

---

# Overall comparison

- The **scatter plot** explains the **relationship between Quantity_Sold and Total_Sales**.
- The **histogram** explains the **distribution of Total_Sales**.
- Together they show that **sales increase with quantity**, but the overall sales distribution is **dominated by many small transactions and a few extremely large transactions**.
- The combination of a **strong positive correlation** and a **right-skewed distribution** suggests that quantity is an important driver of revenue, while product pricing and a small number of high-value orders create additional variability in sales outcomes.

---

# Sprint review

## What I learned

- How to calculate and interpret a **correlation matrix**.
- How to create and analyze **scatter plots** and **histograms**.
- How to identify **positive relationships, skewness, and outliers**.
- How to connect statistical analysis with business insights.
- Why highly correlated features can lead to **multicollinearity** in predictive models.

## Challenges faced

- Handling non-numeric columns while calculating correlations.
- Correcting column name mismatches during plotting.
- Interpreting skewed distributions and outliers.
- Understanding that correlation measures association, not causation.

---

# Conclusion

The analysis shows that **Quantity_Sold is a major driver of Total_Sales**, while **Total_Sales has a right-skewed distribution with a few high-value transactions**. These findings provide useful insights for sales forecasting, pricing strategy, and future predictive modeling.