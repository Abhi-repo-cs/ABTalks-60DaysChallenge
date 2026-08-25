from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "churn"
RANDOM_STATE = 42


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
    }


def remove_highly_correlated_features(X_train, threshold=0.90):
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]
    remaining = [c for c in X_train.columns if c not in to_drop]
    return remaining, to_drop


def main():
    print("=" * 65)
    print("DAY 23 - FEATURE SELECTION")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"Dataset shape: {df.shape}")
    print(f"Original feature count: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 1. Baseline model: all features
    baseline = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test)

    # 2. Feature importance
    importance = pd.Series(
        baseline.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\nFeature importance:")
    print(importance.to_string())

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        header=["importance"]
    )

    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
    plt.close()

    # 3. Correlation analysis
    corr = X_train.corr()

    plt.figure(figsize=(11, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=70,
        ha="right"
    )
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    # 4. Remove highly correlated features
    remaining, correlated_removed = remove_highly_correlated_features(
        X_train,
        threshold=0.90
    )

    # 5. Select features above median importance
    median_importance = importance.median()
    selected_features = [
        f for f in remaining
        if importance[f] >= median_importance
    ]

    # Keep at least 5 features
    if len(selected_features) < 5:
        selected_features = [
            f for f in importance.index if f in remaining
        ][:5]

    # 6. Train model using selected features
    selected_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    selected_model.fit(X_train[selected_features], y_train)
    selected_metrics = evaluate(
        selected_model,
        X_test[selected_features],
        y_test
    )

    # 7. Compare
    comparison = pd.DataFrame(
        [baseline_metrics, selected_metrics],
        index=["All Features", "Selected Features"]
    )
    comparison["Feature Count"] = [
        X_train.shape[1],
        len(selected_features)
    ]

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv")

    plt.figure(figsize=(10, 6))
    comparison[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    ].T.plot(kind="bar")
    plt.title("Model Performance: Before vs After Feature Selection")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close()

    selected_df = pd.DataFrame({
        "selected_feature": selected_features,
        "importance": [importance[f] for f in selected_features]
    }).sort_values("importance", ascending=False)
    selected_df.to_csv(
        OUTPUT_DIR / "selected_features.csv",
        index=False
    )

    report = f"""# Day 23 - Feature Selection Report

## Objective
Identify important predictive signals and remove weak or redundant features.

## Dataset
- Rows: {len(df)}
- Original features: {X.shape[1]}
- Target: `{TARGET}`

## Methods
1. Correlation analysis
2. Random Forest feature importance
3. Removal of highly correlated features (> 0.90)
4. Selection of features with at least median Random Forest importance
5. Before/after model comparison

## Selected Features
{chr(10).join("- " + f for f in selected_features)}

## Correlated Features Removed
{chr(10).join("- " + f for f in correlated_removed) if correlated_removed else "- None"}

## Performance

| Model | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| All Features | {X.shape[1]} | {baseline_metrics["Accuracy"]:.4f} | {baseline_metrics["Precision"]:.4f} | {baseline_metrics["Recall"]:.4f} | {baseline_metrics["F1"]:.4f} | {baseline_metrics["ROC_AUC"]:.4f} |
| Selected Features | {len(selected_features)} | {selected_metrics["Accuracy"]:.4f} | {selected_metrics["Precision"]:.4f} | {selected_metrics["Recall"]:.4f} | {selected_metrics["F1"]:.4f} | {selected_metrics["ROC_AUC"]:.4f} |

## Conclusion
Feature selection reduced the number of inputs from {X.shape[1]} to {len(selected_features)}.
The selected model should be preferred if it maintains comparable performance because it
is simpler, easier to interpret, and uses fewer input variables.
"""
    (OUTPUT_DIR / "comparison_report.md").write_text(report, encoding="utf-8")

    print("\nSelected features:")
    for f in selected_features:
        print("-", f)

    print("\nModel comparison:")
    print(comparison.round(4))

    print("\nAll output files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
