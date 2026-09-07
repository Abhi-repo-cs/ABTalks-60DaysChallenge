from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
rng = np.random.default_rng(36)
dates = pd.date_range("2025-09-01", periods=365, freq="D")
d = np.arange(365)
trend = 42 + 0.075*d
weekly = np.array([-5,-2.5,0,1.5,3,6,7])[dates.dayofweek]
monthly = 4*np.sin(2*np.pi*d/30.5)
quarterly = 3*np.sin(2*np.pi*d/91.25)
noise = rng.normal(0,4,365)
promotion = np.zeros(365)
promo_days = rng.choice(365,10,replace=False)
promotion[promo_days] = rng.uniform(10,28,10)
new_customers = np.clip(trend+weekly+monthly+quarterly+noise+promotion,8,None)
df = pd.DataFrame({"date":dates,"new_customers":np.rint(new_customers).astype(int)})
df.to_csv(BASE_DIR/"customer_growth.csv",index=False)
print("Generated customer_growth.csv")
