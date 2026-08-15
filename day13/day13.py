import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("sales_data_cleaned.csv")

df = df.drop(columns=["Product_ID", "Sale_Date"])


y = df["Total_Sales"]
X = df.drop(columns=["Total_Sales"])


X = pd.get_dummies(X, drop_first=True)

# Ensure all columns are numeric
X = X.astype(float)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(name, model):

    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return {
        "Model": name,
        "Train RMSE": round(train_rmse, 2),
        "Test RMSE": round(test_rmse, 2),
        "Train R2": round(train_r2, 4),
        "Test R2": round(test_r2, 4)
    }


results = []

results.append(
    evaluate_model(
        "Linear Regression",
        LinearRegression()
    )
)

results.append(
    evaluate_model(
        "Ridge Regression",
        Ridge(alpha=1.0)
    )
)

results.append(
    evaluate_model(
        "Lasso Regression",
        Lasso(alpha=0.1)
    )
)

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====\n")
print(results_df)

print("\n===== OVERFITTING ANALYSIS =====\n")

for _, row in results_df.iterrows():

    gap = row["Train R2"] - row["Test R2"]

    print(f"{row['Model']}")
    print(f"Train R2 : {row['Train R2']}")
    print(f"Test R2  : {row['Test R2']}")
    print(f"Gap      : {round(gap, 4)}")

    if gap > 0.1:
        print("Possible Overfitting")
    else:
        print("Good Generalization")

    print()


lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

coefficients = pd.Series(
    lasso.coef_,
    index=X.columns
)

selected = coefficients[coefficients != 0]

print("===== FEATURES SELECTED BY LASSO =====\n")

print(selected.sort_values(
    key=np.abs,
    ascending=False
))

results_df.to_csv("model_comparison.csv", index=False)

print("\nResults saved as model_comparison.csv")