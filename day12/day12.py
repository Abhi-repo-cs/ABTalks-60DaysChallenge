

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("sales_data_cleaned.csv")


print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())



df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])

df["Sale_Year"] = df["Sale_Date"].dt.year
df["Sale_Month_Num"] = df["Sale_Date"].dt.month
df["Sale_Day"] = df["Sale_Date"].dt.day

# Drop original date column
df.drop("Sale_Date", axis=1, inplace=True)


categorical_columns = [
    "Product_ID",
    "Sales_Rep",
    "Region",
    "Product_Category",
    "Customer_Type",
    "Payment_Method",
    "Sales_Channel",
    "Region_and_Sales_Rep",
    "Sale_Month"
]

encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col].astype(str))

X = df.drop("Total_Sales", axis=1)
y = df["Total_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== Model Performance =====")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n===== Feature Coefficients =====")
print(coefficients.sort_values(by="Coefficient", ascending=False))

print("\nIntercept:", model.intercept_)


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Total Sales")
plt.ylabel("Predicted Total Sales")
plt.title("Actual vs Predicted Total Sales")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.show()

errors = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, errors, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Total Sales")
plt.ylabel("Residual Error")
plt.title("Residual Errors")
plt.show()


results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nSample Predictions:")
print(results.head(10))


sample = X.iloc[[0]]
prediction = model.predict(sample)

print("\nPredicted Total Sales for Sample:")
print(prediction[0])


results.to_csv("predicted_total_sales.csv", index=False)

print("\nPredictions saved to predicted_total_sales.csv")