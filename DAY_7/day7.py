import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("sales_data_cleaned.csv")
print(df.corr(numeric_only=True))

#main insights observed
#total sales and Quantidy sold has positive correlation 0.7
#unit price and unit cost has 0.99 correlation causing multicollinearity and may contain the same information

df.plot(kind="scatter",x="Total_Sales",y="Quantity_Sold")
plt.show()

#histogram
df["Total_Sales"].plot(kind='hist')
plt.show()




