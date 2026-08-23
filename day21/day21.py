# ============================================================
# DAY 21 - SPRINT REVIEW & MODEL SELECTION
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

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
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

OUTPUT_DIR = "day21_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("DAY 21 - SPRINT REVIEW & MODEL SELECTION")
print("=" * 75)

print("""
Objective:
Compare classification models and select the most suitable
machine learning system for a real-world problem.
""")


# ============================================================
# 1. CREATE DATASET
# ============================================================

print("=" * 75)
print("1. DATASET")
print("=" * 75)

X, y = make_classification(
    n_samples=3000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,

    # Imbalanced classification problem
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
    f"Dataset size: {df.shape[0]} rows"
)

print(
    f"Features: {len(feature_names)}"
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
    stratify=y,
    random_state=RANDOM_STATE
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
# 5. TRAIN AND COMPARE MODELS
# ============================================================

results = []

trained_models = {}

print("\n")
print("=" * 75)
print("2. MODEL COMPARISON")
print("=" * 75)


for model_name, (model, requires_scaling) in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    if requires_scaling:

        train_data = X_train_scaled
        test_data = X_test_scaled

    else:

        train_data = X_train
        test_data = X_test

    # Train
    model.fit(
        train_data,
        y_train
    )

    trained_models[model_name] = model

    # Predictions
    y_pred = model.predict(
        test_data
    )

    y_prob = model.predict_proba(
        test_data
    )[:, 1]

    # Metrics
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

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "ROC-AUC": roc_auc

    })


# ============================================================
# 6. COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    results
)

print("\n")
print("=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    comparison.round(4).to_string(
        index=False
    )
)


# Save comparison

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 7. RANK MODELS
# ============================================================

# Weighted engineering score
#
# Recall is given higher importance because missing a positive
# case can be costly in an early-warning classification system.

comparison["Engineering Score"] = (

    comparison["Recall"] * 0.30

    + comparison["F1-Score"] * 0.30

    + comparison["ROC-AUC"] * 0.20

    + comparison["Precision"] * 0.10

    + comparison["Accuracy"] * 0.10

)

comparison = comparison.sort_values(
    by="Engineering Score",
    ascending=False
)

comparison["Rank"] = range(
    1,
    len(comparison) + 1
)


# ============================================================
# 8. PRINT RANKING
# ============================================================

print("\n")
print("=" * 75)
print("MODEL RANKING")
print("=" * 75)

ranking_columns = [
    "Rank",
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "Engineering Score"
]

print(
    comparison[
        ranking_columns
    ].round(4).to_string(
        index=False
    )
)


# Save ranking

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_ranking.csv"
    ),
    index=False
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = comparison.iloc[0]["Model"]

best_model_row = comparison.iloc[0]

best_model = trained_models[
    best_model_name
]


print("\n")
print("=" * 75)
print("3. SELECTED MODEL")
print("=" * 75)

print(
    f"\nBEST MODEL: {best_model_name}"
)

print(
    f"Accuracy          : "
    f"{best_model_row['Accuracy']:.4f}"
)

print(
    f"Precision         : "
    f"{best_model_row['Precision']:.4f}"
)

print(
    f"Recall            : "
    f"{best_model_row['Recall']:.4f}"
)

print(
    f"F1-Score          : "
    f"{best_model_row['F1-Score']:.4f}"
)

print(
    f"ROC-AUC           : "
    f"{best_model_row['ROC-AUC']:.4f}"
)

print(
    f"Engineering Score : "
    f"{best_model_row['Engineering Score']:.4f}"
)


# ============================================================
# 10. MODEL STRENGTHS AND WEAKNESSES
# ============================================================

print("\n")
print("=" * 75)
print("4. MODEL STRENGTHS & WEAKNESSES")
print("=" * 75)


model_analysis = {

    "Logistic Regression": {

        "Strength":
        "Simple, fast and highly interpretable.",

        "Weakness":
        "Limited ability to model complex non-linear relationships."

    },

    "Decision Tree": {

        "Strength":
        "Easy to understand and captures non-linear patterns.",

        "Weakness":
        "Can overfit if tree complexity is not controlled."

    },

    "Random Forest": {

        "Strength":
        "Robust ensemble method with good generalization.",

        "Weakness":
        "Less interpretable and more computationally expensive "
        "than a single decision tree."

    },

    "XGBoost": {

        "Strength":
        "Strong predictive performance and effective handling "
        "of complex non-linear relationships.",

        "Weakness":
        "More complex to tune and less directly interpretable "
        "than simpler models."
    }
}


for model_name in models:

    print(
        f"\n{model_name}"
    )

    print(
        "Strength:",
        model_analysis[
            model_name
        ]["Strength"]
    )

    print(
        "Weakness:",
        model_analysis[
            model_name
        ]["Weakness"]
    )


# ============================================================
# 11. BUSINESS-ORIENTED DECISION
# ============================================================

print("\n")
print("=" * 75)
print("5. BUSINESS DECISION")
print("=" * 75)

print(f"""
Selected Model: {best_model_name}

Selection criteria:

30% - Recall
30% - F1-Score
20% - ROC-AUC
10% - Precision
10% - Accuracy

The model was selected using multiple evaluation metrics instead
of accuracy alone.

This approach is appropriate for an imbalanced classification
problem where false negatives can have significant consequences.
""")


# ============================================================
# 12. GENERATE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "best_model_analysis.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 21 - BEST MODEL ANALYSIS\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Selected Model: {best_model_name}\n\n"
    )

    file.write(
        "Evaluation Metrics\n"
    )

    file.write(
        "-" * 40 + "\n"
    )

    file.write(
        f"Accuracy: "
        f"{best_model_row['Accuracy']:.4f}\n"
    )

    file.write(
        f"Precision: "
        f"{best_model_row['Precision']:.4f}\n"
    )

    file.write(
        f"Recall: "
        f"{best_model_row['Recall']:.4f}\n"
    )

    file.write(
        f"F1-Score: "
        f"{best_model_row['F1-Score']:.4f}\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{best_model_row['ROC-AUC']:.4f}\n"
    )

    file.write(
        f"Engineering Score: "
        f"{best_model_row['Engineering Score']:.4f}\n\n"
    )

    file.write(
        "Why this model was selected\n"
    )

    file.write(
        "-" * 40 + "\n"
    )

    file.write(
        "The model was selected based on a combination of "
        "recall, F1-score, ROC-AUC, precision and accuracy. "
        "Recall and F1-score were given higher weights because "
        "the problem involves an imbalanced classification "
        "setting where missing positive cases can be costly.\n\n"
    )

    file.write(
        "Engineering considerations\n"
    )

    file.write(
        "-" * 40 + "\n"
    )

    file.write(
        "The final model should not be selected solely based "
        "on accuracy. A production system should also consider "
        "interpretability, inference speed, scalability, "
        "maintenance requirements and business impact.\n"
    )

print(
    f"\nSaved: {report_path}"
)


# ============================================================
# 13. MODEL COMPARISON GRAPH
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
    "Day 21 - Final Classification Model Comparison"
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
        "final_model_comparison.png"
    ),
    dpi=300
)

plt.close()

print(
    "Saved: day21_outputs/final_model_comparison.png"
)


# ============================================================
# 14. CONFUSION MATRIX OF SELECTED MODEL
# ============================================================

if best_model_name == "Logistic Regression":

    test_data = X_test_scaled

else:

    test_data = X_test


best_predictions = best_model.predict(
    test_data
)

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    f"{best_model_name} - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.yticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "best_model_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 15. WEEK 3 ENGINEERING REFLECTION
# ============================================================

reflection = """
WEEK 3 ENGINEERING REFLECTION

This week focused on moving from simply building machine learning
models to evaluating them as complete engineering systems.

I learned that model development does not end after training.
Different models can perform differently depending on the dataset,
evaluation metric and business requirements.

During the sprint, I compared Logistic Regression, Decision Tree,
Random Forest and XGBoost.

One of the most important lessons was that accuracy alone can be
misleading, especially when the classes are imbalanced. Precision,
recall, F1-score, ROC-AUC and confusion matrices provide a much
better understanding of model behavior.

I also learned that the best machine learning model is not
necessarily the model with the highest accuracy. A production
model must balance predictive performance with interpretability,
scalability, inference cost, maintainability and business impact.

For this sprint, I selected the final model using a weighted
engineering score that gives greater importance to recall and
F1-score.

This week helped me move from thinking about machine learning
as only a modeling problem to thinking about it as an engineering
and decision-making problem.

The next step is to improve model explainability, tune the selected
model and evaluate it on more realistic data.
"""

reflection_path = os.path.join(
    OUTPUT_DIR,
    "week3_engineering_reflection.txt"
)

with open(
    reflection_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        reflection
    )

print(
    f"Saved: {reflection_path}"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 75)
print("DAY 21 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("""
Generated files:

day21_outputs/
│
├── final_model_comparison.csv
├── model_ranking.csv
├── best_model_analysis.txt
├── week3_engineering_reflection.txt
├── final_model_comparison.png
└── best_model_confusion_matrix.png
""")

print(
    f"Final selected model: {best_model_name}"
)

print(
    "\nSprint Review Complete!"
)