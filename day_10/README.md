# Day 10 – Feature Engineering
## 60 Days Data Science Challenge

## Project Overview

This project focuses on **Feature Engineering**, the process of transforming raw data into meaningful features that improve the performance of Machine Learning models.

The dataset contains **1,000 sales records with 14 original features**. The objective was to identify feature types, encode categorical variables, scale numerical variables, create new derived features, and prepare the dataset for machine learning.

---

## Original Dataset

Rows: **1,000**

Columns: **14**

Original features:

- Product_ID
- Sale_Date
- Sales_Rep
- Region
- Sales_Amount
- Quantity_Sold
- Product_Category
- Unit_Cost
- Unit_Price
- Customer_Type
- Discount
- Payment_Method
- Sales_Channel
- Region_and_Sales_Rep

---

# Feature Engineering Steps

## 1. Date Conversion

### Decision

Converted `Sale_Date` from object format to datetime.

### Method

```python
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"], errors="coerce")
```

### Reason

Datetime format enables extraction of useful time-based features such as month and day of the week.

---

## 2. Feature Identification

### Numerical Features

- Sales_Amount
- Quantity_Sold
- Unit_Cost
- Unit_Price
- Discount

### Categorical Features

- Region
- Sales_Rep
- Product_Category
- Customer_Type
- Payment_Method
- Sales_Channel

### Reason

Machine learning algorithms require numerical inputs, so categorical variables must be encoded.

---

## 3. Categorical Encoding

### Decision

Applied **One-Hot Encoding** to categorical features.

### Method

```python
OneHotEncoder(drop="first")
```

### Reason

One-Hot Encoding converts categories into numerical binary features while avoiding multicollinearity by dropping the first category.

---

## 4. Numerical Feature Scaling

### Decision

Applied **StandardScaler** to numerical features.

### Method

```python
StandardScaler()
```

### Features Scaled

- Sales_Amount
- Quantity_Sold
- Unit_Cost
- Unit_Price
- Discount
- Profit_Per_Unit
- Total_Profit
- Discounted_Revenue

### Reason

Scaling ensures all numerical features are on a similar scale, improving model training and optimization.

---

# Derived Features Created

## 1. Profit_Per_Unit

### Formula

```python
Unit_Price - Unit_Cost
```

### Business Value

Represents the profit generated per unit sold.

---

## 2. Total_Profit

### Formula

```python
(Unit_Price - Unit_Cost) * Quantity_Sold
```

### Business Value

Measures the total profit generated from each transaction.

---

## 3. Discounted_Revenue

### Formula

```python
Sales_Amount * (1 - Discount / 100)
```

### Business Value

Captures the actual revenue after discount adjustments.

---

## 4. Sale_Month

Extracted from `Sale_Date`.

### Business Value

Helps identify seasonal sales patterns.

---

## 5. Sale_DayOfWeek

Extracted from `Sale_Date`.

### Business Value

Useful for analyzing weekday versus weekend sales behavior.

---

# Model Readiness Comparison

## Before Feature Engineering

| Issue | Status |
|------|--------|
| Categorical variables | Present |
| Date features | Raw datetime |
| Feature scaling | Not applied |
| Profit features | Not available |
| Time-based features | Not available |

---

## After Feature Engineering

| Improvement | Status |
|------------|--------|
| One-Hot Encoded categorical features | Completed |
| Scaled numerical features | Completed |
| Profit-based features | Created |
| Time-based features | Created |
| ML-ready numerical dataset | Completed |

---

# Output Dataset

The final feature-engineered dataset was saved as:

```text
feature_engineered_sales_data.csv
```

---

# Final Dataset Characteristics

- Original features: **14**
- Derived features: **5**
- Encoded categorical features: **Multiple binary columns**
- Final dataset: **Fully numerical and machine-learning ready**

---

# Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

# Key Learning

This project demonstrated that **Feature Engineering is one of the most impactful stages of the machine learning pipeline**. Creating meaningful features such as profitability and time-based variables often provides more predictive power than simply changing machine learning algorithms.

---

