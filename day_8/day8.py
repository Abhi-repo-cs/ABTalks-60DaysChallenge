import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("sales_data_cleaned.csv")
#summarize a dataset 
df.info()
print("The descriptive summary of a dataset are: ", df.describe(include="all"))

#visualizations
sns.histplot(df['Total_Sales'], kde=True)
plt.show()


#box plot to detect outliers
plt.figure(figsize=(8,4))
sns.boxplot(x=df["Total_Sales"])
plt.title("Box plot of total sales")
plt.show()

#relationship bw quantity and sales
plt.figure(figsize=(8,5))
sns.scatterplot(data=df,
                x="Quantity_Sold",
                y="Total_Sales")
plt.title("Quantity sold vs total sales")
plt.show()

#correalation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation matrix")
plt.show()

#sales by region
region_sales = df.groupby("Region")["Total_Sales"].sum().sort_values()

region_sales.plot(kind="barh", figsize=(8,5))
plt.title("Total sales by region")
plt.show()

#monthly sales
monthly_sales = df.groupby("Sale_Month")["Total_Sales"].sum()

monthly_sales.plot(kind="line", marker="o", figsize=(8,5))
plt.title("Monthly sales trend")
plt.ylabel("Total sales")
plt.show()