# Sales Dataset Cleaning Report

## Project Overview
This project involved cleaning and preprocessing a sales dataset containing **1,000 records and 14 columns**. The goal was to improve data quality, ensure consistency, and prepare the dataset for exploratory data analysis (EDA) and machine learning.

---

## Original Dataset

- Rows: **1,000**
- Columns: **14**

Columns included:

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

## Data Cleaning Steps

### 1. Missing Value Check

**Method**

```python
df.isna().sum()
```

**Decision**

- Checked all columns for null values.
- No missing values were found.

**Action Taken**

- No imputation or deletion was required.

---

### 2. Duplicate Record Check

**Method**

```python
df.duplicated().sum()
```

**Decision**

- Checked for duplicate rows across the dataset.
- No duplicate records were found.

**Action Taken**

- No duplicate rows were removed.

---

### 3. Date Format Standardization

**Method**

```python
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], errors='coerce')
```

**Decision**

- Converted `Sale_Date` to `datetime` format.

**Action Taken**

- Invalid dates would have been converted to `NaT`.
- No formatting issues remained after conversion.

---

### 4. Numeric Data Type Validation

**Columns Checked**

- Sales_Amount
- Quantity_Sold
- Unit_Cost
- Unit_Price
- Discount

**Method**

```python
pd.to_numeric(column, errors='coerce')
```

**Decision**

- Ensured all numeric columns were stored using numeric data types.

**Action Taken**

- Converted values to numeric format where necessary.

---

### 5. Text Standardization

**Columns Standardized**

- Region
- Sales_Rep
- Customer_Type
- Payment_Method
- Sales_Channel

**Method**

```python
.str.strip().str.title()
```

**Decision**

- Removed leading and trailing spaces.
- Standardized capitalization.

**Action Taken**

- Ensured consistent categorical values.

---

## Quality Checks Performed

- Missing values
- Duplicate rows
- Date formatting
- Numeric formatting
- Text consistency
- Column validation

---

## Final Dataset Status

| Check | Result |
|------|--------|
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Invalid Date Formats | 0 |
| Numeric Type Issues | Resolved |
| Text Formatting Issues | Standardized |

---

## Output

The cleaned dataset was saved as:

```text
cleaned_sales_data.csv
```

---

## Tools Used

- Python
- Pandas

---

## Summary

The dataset was already complete with **no missing values and no duplicate records**. Cleaning focused on **format standardization, data type validation, and categorical consistency**, resulting in a clean dataset suitable for EDA, visualization, statistical analysis, and predictive modeling.