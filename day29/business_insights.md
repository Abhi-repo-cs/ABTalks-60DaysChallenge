# Customer Segmentation — Business Insight Report

## 1. Objective

The objective of this project is to identify meaningful customer groups using **K-Means clustering** and translate the resulting segments into actionable business strategies.

The analysis uses customer demographic and behavioral variables, with the primary clustering features being:

- Age
- Annual Income (k$)
- Spending Score (1-100)

## 2. Methodology

The customer data was standardized before applying K-Means so that variables with different scales would contribute appropriately to the clustering process.

Different cluster counts from **K = 2 to K = 10** were evaluated using:

- Elbow Method
- Silhouette Score

The Elbow curve shows a substantial reduction in inertia up to around **K = 5**, after which the improvement becomes more gradual.

The Silhouette Score reaches its highest value at approximately **K = 5 (0.558)** and decreases for larger cluster counts.

Based on both measures, **5 clusters** was selected for the final segmentation model.

## 3. Customer Segment Analysis

The final K-Means visualization shows five clearly distinguishable customer groups based primarily on income and spending behavior.

> Note: Cluster numbers are model-generated labels. Their numeric order does not imply a business ranking.

### Cluster 0 — Lower-Income, High-Spending Customers

**Observed characteristics:**
- Relatively lower annual income
- High spending scores
- Predominantly younger customers in the age visualization

**Business interpretation:**

These customers show strong engagement and spending behavior despite having comparatively lower income.

**Recommended strategies:**
- Affordable premium-looking products
- Discounts and promotional bundles
- Loyalty rewards
- Limited-time offers
- Social-media-oriented campaigns

**Business opportunity:**

This segment has strong engagement and can be encouraged to increase purchase frequency and customer lifetime value through loyalty-focused offers.

---

### Cluster 1 — High-Income, Low-Spending Customers

**Observed characteristics:**
- High annual income
- Low spending scores
- Generally older customers

**Business interpretation:**

These customers have significant purchasing power but are not currently spending heavily.

**Recommended strategies:**
- Personalized product recommendations
- Targeted introductory offers
- Premium product demonstrations
- Personalized email campaigns
- Customer feedback surveys

**Business opportunity:**

This is a potentially valuable conversion segment. The objective should be to understand why customers with high purchasing power have relatively low engagement and remove barriers to purchase.

---

### Cluster 2 — Lower-Income, Low-Spending Customers

**Observed characteristics:**
- Lower annual income
- Low spending scores
- Mostly older customers

**Business interpretation:**

These customers show relatively low purchasing activity and lower spending potential.

**Recommended strategies:**
- Budget-friendly products
- Simple promotional campaigns
- Essential-product offers
- Cost-efficient digital marketing

**Business opportunity:**

Marketing investment should remain cost-effective for this group. Broad, low-cost campaigns may be more appropriate than expensive personalized campaigns.

---

### Cluster 3 — High-Income, High-Spending Customers

**Observed characteristics:**
- High annual income
- High spending scores
- Strong customer engagement
- Relatively younger/middle-aged profile in the visualization

**Business interpretation:**

This is the strongest high-value customer segment in the analysis.

**Recommended strategies:**
- VIP or premium loyalty programs
- Exclusive product launches
- Personalized recommendations
- Early access to products
- Premium bundles
- Referral programs

**Business opportunity:**

These customers should receive strong retention efforts because losing highly engaged, high-spending customers can have a significant impact on revenue.

---

### Cluster 4 — Middle-Income, Moderate-Spending Customers

**Observed characteristics:**
- Middle-range annual income
- Moderate spending scores
- Middle-aged customer profile

**Business interpretation:**

These customers represent a stable middle segment with potential to move toward higher engagement.

**Recommended strategies:**
- Cross-selling
- Product recommendations
- Loyalty points
- Personalized discounts
- Bundle offers

**Business opportunity:**

This segment can potentially be developed into a higher-value group by increasing purchase frequency and average customer engagement.

## 4. Key Business Insights

### 1. Income does not completely determine spending

The clustering demonstrates that customers with similar income levels can have very different spending behaviors.

For example, the analysis identifies both:

- High-income, high-spending customers
- High-income, low-spending customers

This shows why businesses should consider behavioral variables rather than relying only on income or demographics.

### 2. High-income, low-spending customers are an important opportunity

Customers in this segment have purchasing power but relatively low spending activity.

Instead of treating them as low-value customers, businesses should investigate their preferences, barriers, product fit, and engagement patterns.

### 3. High-income, high-spending customers deserve retention priority

The high-income/high-spending segment represents customers who are already highly engaged.

Retention strategies, VIP benefits, personalized experiences, and early access can help protect their customer lifetime value.

### 4. Lower-income, high-spending customers can be highly engaged

Income alone should not be used to determine customer importance.

The lower-income/high-spending segment demonstrates that customers with lower purchasing power can still show strong engagement and should receive appropriate loyalty and retention strategies.

### 5. Segmentation enables targeted marketing

Instead of sending the same campaign to every customer, the business can design different strategies for each segment.

This can improve:

- Marketing relevance
- Customer engagement
- Retention
- Campaign efficiency
- Cross-selling opportunities

## 5. Limitations

K-Means clustering is sensitive to the selected features, scaling, initialization, and number of clusters.

The segments should therefore be treated as analytical groups rather than absolute customer categories.

The analysis would become stronger with additional real-world variables such as:

- Purchase frequency
- Customer lifetime value
- Recency
- Total spending
- Product category preferences
- Website engagement
- Loyalty membership
- Discount usage

These variables could provide a more complete behavioral segmentation.

## 6. Conclusion

The K-Means analysis successfully identified **five customer segments** with distinct income, spending, and demographic characteristics.

The main lesson is that the value of customer segmentation comes not only from creating clusters, but from translating those clusters into **specific business actions**.

The resulting segments can support personalized marketing, customer retention, product recommendations, and more efficient allocation of marketing resources.
