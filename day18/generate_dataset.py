import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

transaction_amount = np.random.lognormal(
    mean=4.2,
    sigma=1.0,
    size=n
)

transaction_hour = np.random.randint(0, 24, n)

account_age = np.random.randint(10, 3000, n)

previous_transactions = np.random.randint(1, 500, n)

failed_transactions = np.random.poisson(2, n)

device_trust = np.random.uniform(0, 100, n)

location_change = np.random.binomial(1, 0.15, n)

international = np.random.binomial(1, 0.20, n)

transaction_frequency = np.random.poisson(5, n)

# Create fraud probability
fraud_score = (
    0.0008 * transaction_amount
    + 0.8 * location_change
    + 0.7 * international
    + 0.12 * failed_transactions
    + 0.08 * transaction_frequency
    - 0.015 * device_trust
    - 0.0001 * account_age
)

probability = 1 / (1 + np.exp(-fraud_score + 2.5))

fraud = np.random.binomial(1, probability)

df = pd.DataFrame({
    "Transaction_Amount": transaction_amount.round(2),
    "Transaction_Hour": transaction_hour,
    "Account_Age_Days": account_age,
    "Previous_Transactions": previous_transactions,
    "Failed_Transactions": failed_transactions,
    "Device_Trust_Score": device_trust.round(2),
    "Location_Change": location_change,
    "International_Transaction": international,
    "Transaction_Frequency": transaction_frequency,
    "Fraud": fraud
})

df.to_csv("fraud_detection.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print("\nFraud distribution:")
print(df["Fraud"].value_counts())