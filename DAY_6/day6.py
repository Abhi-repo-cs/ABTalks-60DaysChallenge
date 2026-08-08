import pandas as pd

# Load the dataset
df = pd.read_csv("sales_data.csv")
print(df.columns.tolist())

# Remove empty values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Fix data types
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"], errors="coerce")
df["Quantity_Sold"] = pd.to_numeric(df["Quantity_Sold"], errors="coerce")
df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

# Create 2 new features
df["Total_Sales"] = df["Quantity_Sold"] * df["Unit_Price"]
df["Sale_Month"] = df["Sale_Date"].dt.month_name()

# Display the cleaned dataset
print(df.head())

# Check data types
print(df.dtypes)

# Save the cleaned dataset
df.to_csv("sales_data_cleaned.csv", index=False)