# ============================================================
# DAY 20 - WHY ACCURACY ALONE CAN MISLEAD YOU
# Model Evaluation
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

OUTPUT_DIR = "outputs"
CONFUSION_DIR = os.path.join(
    OUTPUT_DIR,
    "confusion_matrices"
)

os.makedirs(
    CONFUSION_DIR,
    exist_ok=True
)


# ============================================================
# 1. CREATE DATASET
# ============================================================

print("=" * 70)
print("DAY 20 - MODEL EVALUATION")
print("=" * 70)

print("\nCreating classification dataset...")

X, y = make_classification(
    n_samples=3000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,

    # Imbalanced dataset
    weights=[0.80, 0.20],

    random_state=RANDOM_STATE
)

feature_names = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
    "feature_8",
    "feature_9",
    "feature_10"
]

df = pd.DataFrame(
    X,
    columns=feature_names
)

df["target"] = y

os.makedirs(
    "data",
    exist_ok=True
)

df.to_csv(
    "data/classification_data.csv",
    index=False
)

print(
    f"Dataset created: {df.shape[0]} rows, "
    f"{df.shape[1] - 1} features"
)

print("\nClass distribution:")
print(
    df["target"].value_counts()
)


# ============================================================
# 2. TRAIN TEST SPLIT
# ============================================================

X = df.drop(
    "target",
    axis=1
)

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 3. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        True
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=6,
            random_state=RANDOM_STATE
        ),
        False
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE
        ),
        False
    ),

    "XGBoost": (
        XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
        False
    )
}


# ============================================================
# 5. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

roc_data = {}

print("\n")
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


for model_name, (model, requires_scaling) in models.items():

    print("\n" + "=" * 70)
    print(model_name.upper())
    print("=" * 70)

    # --------------------------------------------------------
    # Select data
    # --------------------------------------------------------

    if requires_scaling:

        train_data = X_train_scaled
        test_data = X_test_scaled

    else:

        train_data = X_train
        test_data = X_test

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        train_data,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        test_data
    )

    y_prob = model.predict_proba(
        test_data
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Negative",
            "Positive"
        ],
        yticklabels=[
            "Negative",
            "Positive"
        ]
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".png"
    )

    plt.savefig(
        os.path.join(
            CONFUSION_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # ROC curve data
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }


# ============================================================
# 6. CREATE COMPARISON DATAFRAME
# ============================================================

comparison = pd.DataFrame(
    results
)

comparison = comparison.sort_values(
    by="F1-Score",
    ascending=False
)

print("\n")
print("=" * 70)
print("FINAL METRICS COMPARISON")
print("=" * 70)

print(
    comparison.round(4).to_string(
        index=False
    )
)


# ============================================================
# 7. SAVE METRICS TABLE
# ============================================================

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "metrics_comparison.csv"
    ),
    index=False
)

print(
    "\nSaved: outputs/metrics_comparison.csv"
)


# ============================================================
# 8. METRICS COMPARISON CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

plot_data = comparison.set_index(
    "Model"
)[metrics]

ax = plot_data.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Classification Model Performance Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    rotation=20
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "metrics_comparison.png"
    ),
    dpi=300
)

plt.close()

print(
    "Saved: outputs/metrics_comparison.png"
)


# ============================================================
# 9. ROC CURVE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 7)
)

for model_name, data in roc_data.items():

    plt.plot(
        data["fpr"],
        data["tpr"],
        label=(
            f"{model_name} "
            f"(AUC = {data['auc']:.3f})"
        )
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "ROC Curve Comparison"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curves.png"
    ),
    dpi=300
)

plt.close()

print(
    "Saved: outputs/roc_curves.png"
)


# ============================================================
# 10. BEST MODEL
# ============================================================

best_model = comparison.iloc[0]

print("\n")
print("=" * 70)
print("BEST MODEL BASED ON F1-SCORE")
print("=" * 70)

print(
    f"Model     : {best_model['Model']}"
)

print(
    f"Accuracy  : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision : {best_model['Precision']:.4f}"
)

print(
    f"Recall    : {best_model['Recall']:.4f}"
)

print(
    f"F1-Score  : {best_model['F1-Score']:.4f}"
)

print(
    f"ROC-AUC   : {best_model['ROC-AUC']:.4f}"
)


# ============================================================
# 11. WHY ACCURACY CAN MISLEAD
# ============================================================

print("\n")
print("=" * 70)
print("WHY ACCURACY ALONE CAN MISLEAD")
print("=" * 70)

print("""
Accuracy represents the percentage of all predictions that are correct.

However, accuracy can be misleading when classes are imbalanced.

Example:

1000 total samples
950 negative samples
50 positive samples

A model predicting every sample as negative would achieve:

Accuracy = 950 / 1000 = 95%

But:

Recall for positive class = 0%

The model completely fails to identify positive cases.

Therefore, classification systems should also consider:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
""")


# ============================================================
# 12. MODEL STRENGTHS AND WEAKNESSES
# ============================================================

print("=" * 70)
print("MODEL STRENGTHS AND WEAKNESSES")
print("=" * 70)

print("""
Logistic Regression
-------------------
Strength:
- Simple and interpretable
- Fast to train
- Good baseline model

Weakness:
- Assumes a relatively simple decision boundary


Decision Tree
-------------
Strength:
- Easy to interpret
- Captures non-linear relationships

Weakness:
- Can overfit without proper constraints


Random Forest
-------------
Strength:
- Robust ensemble method
- Reduces overfitting compared with a single tree
- Handles non-linear relationships well

Weakness:
- Less interpretable than a single decision tree
- Larger computational cost


XGBoost
-------
Strength:
- Powerful gradient boosting algorithm
- Often achieves strong predictive performance
- Handles complex relationships

Weakness:
- More hyperparameters
- More difficult to interpret
- Requires careful tuning
""")


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("DAY 20 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Generated:

outputs/
|
|-- metrics_comparison.csv
|-- metrics_comparison.png
|-- roc_curves.png
|
`-- confusion_matrices/
    |-- logistic_regression.png
    |-- decision_tree.png
    |-- random_forest.png
    `-- xgboost.png
""")

