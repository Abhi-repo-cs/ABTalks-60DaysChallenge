import pandas as pd
df=pd.read_csv("sales_data.csv")
print(df.isna().any(axis=1))
missing = pd.DataFrame({
    "Missing Values": df.isna().sum(),
    "Percentage": (df.isna().sum() / len(df)) * 100
})
print(missing)
#there are no missing values in the dataframe

print(df.duplicated())
print("Duplicate rows:", df.duplicated().sum())
#there is no duplicates in the dataset

print(df.columns)

df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], errors='coerce')
df.to_csv("cleaned_sales_data.csv", index=False)
