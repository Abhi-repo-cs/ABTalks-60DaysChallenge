import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("customer_transactions.csv")

# Select behavioral features
features = [
    "total_spending",
    "transaction_count",
    "average_transaction",
    "purchase_frequency"
]

X = df[features]

# Build anomaly detection model
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["anomaly"] = model.fit_predict(X)

# Convert prediction into readable labels
df["status"] = df["anomaly"].map({
    1: "Normal",
    -1: "Suspicious"
})

# Display suspicious customers
suspicious = df[df["status"] == "Suspicious"]

print("Total customers:", len(df))
print("Suspicious customers:", len(suspicious))
print("\nSuspicious Customer Records:")
print(suspicious)

# Visualization
plt.figure(figsize=(10, 6))

normal = df[df["status"] == "Normal"]
anomalies = df[df["status"] == "Suspicious"]

plt.scatter(
    normal["transaction_count"],
    normal["total_spending"],
    label="Normal"
)

plt.scatter(
    anomalies["transaction_count"],
    anomalies["total_spending"],
    label="Suspicious",
    marker="x",
    s=100
)

plt.xlabel("Transaction Count")
plt.ylabel("Total Spending")
plt.title("Customer Anomaly Detection")
plt.legend()
plt.grid(True)

plt.savefig("anomaly_detection.png")
plt.show()