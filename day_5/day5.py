import pandas as pd
data=pd.read_csv("sales_data.csv")
print("Information of the Dataset: ",data.info)
print("\nShape of the Data: ", data.shape)
print("\nColumns in the Data: ", data.columns)
print("\nFirst 10 entries in the Dataset: " ,data.head(10))
target="Sales_Amount"
print("The target variable for the problem statement is: ",target)