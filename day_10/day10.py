
# Import libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("cleaned_sales_data.csv")

print("="*50)
print("Dataset Loaded")
print("="*50)

print(df.head())

print("\nDataset Shape:", df.shape)



print("\nData Types")
print(df.dtypes)


df["Sale_Date"] = pd.to_datetime(df["Sale_Date"], errors="coerce")


numerical_features = [
    "Sales_Amount",
    "Quantity_Sold",
    "Unit_Cost",
    "Unit_Price",
    "Discount"
]

categorical_features = [
    "Region",
    "Sales_Rep",
    "Product_Category",
    "Customer_Type",
    "Payment_Method",
    "Sales_Channel"
]

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)



# Feature 1: Profit per unit
df["Profit_Per_Unit"] = df["Unit_Price"] - df["Unit_Cost"]

# Feature 2: Total Profit
df["Total_Profit"] = (
    (df["Unit_Price"] - df["Unit_Cost"])
    * df["Quantity_Sold"]
)

# Feature 3: Discounted Revenue
df["Discounted_Revenue"] = (
    df["Sales_Amount"]
    * (1 - df["Discount"] / 100)
)

# Feature 4: Sale Month
df["Sale_Month"] = df["Sale_Date"].dt.month

# Feature 5: Sale Day of Week
df["Sale_DayOfWeek"] = df["Sale_Date"].dt.dayofweek

print("\nDerived Features Created")



encoder = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

encoded_array = encoder.fit_transform(df[categorical_features])

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(categorical_features),
    index=df.index
)

# Remove original categorical columns
df = df.drop(columns=categorical_features)

# Combine encoded columns
df = pd.concat([df, encoded_df], axis=1)

print("\nCategorical Features Encoded")



scaler = StandardScaler()

scale_columns = [
    "Sales_Amount",
    "Quantity_Sold",
    "Unit_Cost",
    "Unit_Price",
    "Discount",
    "Profit_Per_Unit",
    "Total_Profit",
    "Discounted_Revenue"
]

df[scale_columns] = scaler.fit_transform(df[scale_columns])

print("\nNumerical Features Scaled")



df = df.drop(columns=["Sale_Date"])



print("\n" + "="*50)
print("MODEL READINESS COMPARISON")
print("="*50)

print("Before Feature Engineering")

original_df = pd.read_csv("cleaned_sales_data.csv")

print("Shape:", original_df.shape)

print("Categorical Columns:")
print(original_df.select_dtypes(include="object").columns.tolist())

print("\nAfter Feature Engineering")

print("Shape:", df.shape)

print("Remaining Categorical Columns:")
print(df.select_dtypes(include="object").columns.tolist())

print("\nFeature Count Increased From")

print(original_df.shape[1], "to", df.shape[1])


print("\nMissing Values")
print(df.isna().sum().sum())



output_file = "feature_engineered_sales_data.csv"

df.to_csv(output_file, index=False)

print("\nDataset Saved Successfully")
print("File:", output_file)



print("\nFinal Dataset Shape:", df.shape)

print("\nFirst Five Rows")
print(df.head())

print("\nFinal Data Types")
print(df.dtypes)