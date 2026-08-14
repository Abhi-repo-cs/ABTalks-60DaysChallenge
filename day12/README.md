# Day 12 – Regression Modeling: Total Sales Prediction

## Overview

This project demonstrates **Linear Regression** for predicting **Total Sales** using historical sales data. The model learns the relationship between sales-related features and the target variable (`Total_Sales`) and evaluates prediction accuracy using regression metrics.

## Objective

- Train a Linear Regression model
- Predict continuous numerical values
- Interpret feature coefficients
- Visualize prediction performance
- Evaluate model accuracy

## Dataset Features

The dataset includes the following columns:

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
- Total_Sales (Target)
- Sale_Month

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Workflow

1. Load and inspect the dataset
2. Convert `Sale_Date` into numerical date features
3. Encode categorical variables
4. Split the dataset into training and testing sets
5. Train a Linear Regression model
6. Predict Total Sales
7. Evaluate performance using MAE, MSE, RMSE, and R² score
8. Visualize prediction results and residual errors

## Model Evaluation

The project uses the following regression metrics:

- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

## Visualizations

- Actual vs Predicted Total Sales
- Residual Error Plot

## Repository Structure

```
day12/
│
├── day12.py
├── sales_data_cleaned.csv
├── predicted_total_sales.csv
├── predicted_Total_Sales.png
├── visualization.png
└── README.md
```

## Results

The Linear Regression model predicts Total Sales and provides interpretable coefficients that show how each feature influences the prediction. The evaluation metrics help measure prediction accuracy and overall model performance.

## Learning Outcomes

- Understanding regression modeling
- Feature preprocessing
- Categorical data encoding
- Model training and testing
- Coefficient interpretation
- Prediction error analysis
- Regression visualization

## Author

Day 12 of my Data Science learning journey – exploring regression modeling for numerical prediction and business forecasting.