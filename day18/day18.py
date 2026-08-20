import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------

df = pd.read_csv("fraud_detection.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nFraud Distribution:")
print(df["Fraud"].value_counts())

# ---------------------------------------------------
# 2. Separate Features and Target
# ---------------------------------------------------

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# ---------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# 4. Decision Tree Model
# ---------------------------------------------------

decision_tree = DecisionTreeClassifier(
    max_depth=8,
    random_state=42
)

decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions, zero_division=0)
dt_recall = recall_score(y_test, dt_predictions, zero_division=0)
dt_f1 = f1_score(y_test, dt_predictions, zero_division=0)

# ---------------------------------------------------
# 5. Random Forest Model
# ---------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, zero_division=0)
rf_recall = recall_score(y_test, rf_predictions, zero_division=0)
rf_f1 = f1_score(y_test, rf_predictions, zero_division=0)

# ---------------------------------------------------
# 6. Performance Comparison
# ---------------------------------------------------

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_accuracy, rf_accuracy],
    "Precision": [dt_precision, rf_precision],
    "Recall": [dt_recall, rf_recall],
    "F1 Score": [dt_f1, rf_f1]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison)

# ---------------------------------------------------
# 7. Classification Reports
# ---------------------------------------------------

print("\n========== DECISION TREE ==========")
print(classification_report(
    y_test,
    dt_predictions,
    zero_division=0
))

print("\n========== RANDOM FOREST ==========")
print(classification_report(
    y_test,
    rf_predictions,
    zero_division=0
))

# ---------------------------------------------------
# 8. Confusion Matrix - Random Forest
# ---------------------------------------------------

cm = confusion_matrix(y_test, rf_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

disp.plot()
plt.title("Random Forest - Fraud Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ---------------------------------------------------
# 9. Feature Importance
# ---------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ---------------------------------------------------
# 10. Model Comparison Chart
# ---------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    comparison.iloc[0][metrics],
    width,
    label="Decision Tree"
)

plt.bar(
    x + width / 2,
    comparison.iloc[1][metrics],
    width,
    label="Random Forest"
)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Decision Tree vs Random Forest")
plt.legend()

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# ---------------------------------------------------
# 11. Robustness Test
# ---------------------------------------------------

print("\n========== ROBUSTNESS TEST ==========")

for seed in [10, 20, 30, 40, 50]:

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train_r, y_train_r)

    predictions = model.predict(X_test_r)

    score = f1_score(
        y_test_r,
        predictions,
        zero_division=0
    )

    print(
        f"Random State {seed}: "
        f"F1 Score = {score:.4f}"
    )

print("\nFraud detection analysis completed successfully.")