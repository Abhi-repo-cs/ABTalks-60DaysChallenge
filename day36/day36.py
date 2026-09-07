from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR/"customer_growth.csv",parse_dates=["date"])
df = df.sort_values("date").set_index("date").asfreq("D")
series = df["new_customers"].fillna(0).astype(float)

plt.figure(figsize=(12,6))
plt.plot(series,label="Daily New Customers",alpha=.45)
plt.plot(series.rolling(7).mean(),label="7-Day Moving Average",linewidth=2)
plt.plot(series.rolling(30).mean(),label="30-Day Moving Average",linewidth=2)
plt.title("Historical Customer Growth Trend"); plt.xlabel("Date"); plt.ylabel("New Customers")
plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
plt.savefig(BASE_DIR/"trend_visualization.png",dpi=150); plt.close()

model = ExponentialSmoothing(series,trend="add",seasonal="add",seasonal_periods=7,
                             initialization_method="estimated")
forecast = model.fit(optimized=True).forecast(30).clip(lower=0)
out = pd.DataFrame({"date":forecast.index,
                    "predicted_new_customers":np.round(forecast.values,2),
                    "model":"Holt-Winters Exponential Smoothing"})
out.to_csv(BASE_DIR/"forecast_output.csv",index=False)

plt.figure(figsize=(12,6))
plt.plot(series,label="Historical New Customers",alpha=.65)
plt.plot(forecast,label="30-Day Forecast",linewidth=2)
plt.axvline(series.index[-1],linestyle="--",label="Forecast Start")
plt.title("Customer Growth Forecast - Next 30 Days"); plt.xlabel("Date"); plt.ylabel("New Customers")
plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
plt.savefig(BASE_DIR/"forecast_30_days.png",dpi=150); plt.close()
print(out.to_string(index=False))
