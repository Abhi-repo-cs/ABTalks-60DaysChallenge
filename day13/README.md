# Day 13 – Preventing Models from Memorizing Data (Overfitting & Regularization)

## Project Overview

This project demonstrates how machine learning models can overfit training data and how **Ridge Regression (L2 Regularization)** and **Lasso Regression (L1 Regularization)** improve generalization performance.

The objective is to compare a baseline **Linear Regression** model with regularized regression models and analyze their behavior on unseen test data.

---

## Objectives

- Train a baseline Linear Regression model
- Train Ridge Regression and Lasso Regression models
- Compare training and testing performance
- Identify signs of overfitting
- Analyze the impact of regularization
- Save model comparison results

---

## Dataset

The dataset contains sales transactions with product, pricing, customer, and regional information.

### Features

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
- Sale_Month

### Target Variable

- **Total_Sales**

---

## Project Workflow

### 1. Data Preprocessing

- Loaded the dataset
- Removed unnecessary columns
- Encoded categorical variables using one-hot encoding
- Split data into training and testing sets
- Standardized numerical features

### 2. Model Training

The following models were trained:

- Linear Regression
- Ridge Regression
- Lasso Regression

### 3. Model Evaluation

Performance was measured using:

- RMSE (Root Mean Squared Error)
- R² Score

Training and testing performance were compared to detect overfitting.

---

## Model Comparison

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
|------|------:|------:|------:|------:|
| Linear Regression | XX | XX | XX | XX |
| Ridge Regression | XX | XX | XX | XX |
| Lasso Regression | XX | XX | XX | XX |

*(Replace XX with your actual results.)*

---

## Overfitting Analysis

A model is considered to be overfitting when:

- Training R² is significantly higher than testing R²
- Training error is much lower than testing error

### Observations

#### Linear Regression

- Achieved the highest training accuracy
- Showed a larger train-test performance gap
- More prone to overfitting

#### Ridge Regression

- Reduced coefficient magnitudes
- Improved generalization
- Produced more stable predictions

#### Lasso Regression

- Performed feature selection
- Eliminated less important variables
- Created a simpler and more interpretable model

---

## Key Learnings

- High training accuracy does not guarantee good real-world performance.
- Regularization reduces model complexity.
- Ridge shrinks coefficients while retaining all features.
- Lasso can remove irrelevant features entirely.
- Generalization is more important than memorizing training data.

---

## Files Included

```
day13/
│
├── day13.py
├── sales_data_cleaned.csv
├── model_comparison.csv
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

## Results

The experiment showed that regularized models generally achieved a better balance between training and testing performance, making them more suitable for production machine learning systems.

---

## Author

**Abhishek**

Day 13 of the **60 Days Data Science Challenge**