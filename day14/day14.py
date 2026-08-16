# ============================================================
# DAY 14 - DATA SCIENCE CHALLENGE
# Adapting Models to Changing Constraints
#
# Experiment:
# 1. Train baseline model
# 2. Measure performance
# 3. Remove important feature: Sales_Amount
# 4. Retrain model
# 5. Compare performance
# 6. Analyze feature importance
# 7. Save results and models
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_FILE = "sales_data_cleaned.csv"

TARGET = "Total_Sales"

FEATURE_TO_REMOVE = "Sales_Amount"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# 3. CHECK DATASET
# ============================================================

print("=" * 60)
print("DAY 14 - DATA SCIENCE CHALLENGE")
print("=" * 60)

print("\nChecking dataset...")

if not os.path.exists(DATA_FILE):
    print("\nERROR:")
    print(f"Dataset '{DATA_FILE}' was not found.")
    print("\nFiles available in this folder:")

    for file in os.listdir("."):
        print(" -", file)

    print(
        "\nChange DATA_FILE at the top of the program "
        "to your actual CSV filename."
    )

    exit()


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 5. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumn names:")

for column in df.columns:
    print(" -", column)


print("\nData types:")
print(df.dtypes)


print("\nMissing values:")

print(df.isnull().sum())


print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\nDuplicates removed:",
      before_duplicates - after_duplicates)


# ============================================================
# 7. CHECK TARGET
# ============================================================

if TARGET not in df.columns:

    print(
        f"\nERROR: Target column '{TARGET}' "
        "does not exist."
    )

    print("\nAvailable columns:")
    print(df.columns.tolist())

    exit()


# ============================================================
# 8. DATE PROCESSING
# ============================================================

if "Sale_Date" in df.columns:

    print("\nProcessing Sale_Date...")

    df["Sale_Date"] = pd.to_datetime(
        df["Sale_Date"],
        errors="coerce"
    )

    df["Sale_Year"] = df["Sale_Date"].dt.year

    df["Sale_Month_Num"] = df["Sale_Date"].dt.month

    df["Sale_Day"] = df["Sale_Date"].dt.day

    df["Sale_DayOfWeek"] = (
        df["Sale_Date"].dt.dayofweek
    )

    df = df.drop(
        columns=["Sale_Date"]
    )

    print("Date features created successfully.")


# ============================================================
# 9. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 60)
print("FEATURE / TARGET SEPARATION")
print("=" * 60)

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


print("\nTarget column:")
print(TARGET)

print("\nNumber of features:")
print(X.shape[1])

print("\nFeatures:")

for column in X.columns:
    print(" -", column)


# ============================================================
# 10. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ============================================================
# 11. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


print("\nNumerical features:")
print(numeric_features)


print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 12. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 13. BASELINE RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL")
print("=" * 60)


baseline_model = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            baseline_model
        )
    ]
)


# ============================================================
# 14. TRAIN BASELINE MODEL
# ============================================================

print("\nTraining baseline model...")

baseline_pipeline.fit(
    X_train,
    y_train
)

print("Baseline model trained successfully.")


# ============================================================
# 15. BASELINE PREDICTIONS
# ============================================================

y_pred_baseline = baseline_pipeline.predict(
    X_test
)


# ============================================================
# 16. BASELINE METRICS
# ============================================================

mae_baseline = mean_absolute_error(
    y_test,
    y_pred_baseline
)

mse_baseline = mean_squared_error(
    y_test,
    y_pred_baseline
)

rmse_baseline = np.sqrt(
    mse_baseline
)

r2_baseline = r2_score(
    y_test,
    y_pred_baseline
)


print("\nBaseline Performance:")

print(
    f"MAE  : {mae_baseline:.4f}"
)

print(
    f"MSE  : {mse_baseline:.4f}"
)

print(
    f"RMSE : {rmse_baseline:.4f}"
)

print(
    f"R2   : {r2_baseline:.4f}"
)


# ============================================================
# 17. BASELINE FEATURE IMPORTANCE
# ============================================================

feature_names = (
    baseline_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


importances = (
    baseline_pipeline
    .named_steps["model"]
    .feature_importances_
)


feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTop 15 baseline features:")

print(
    feature_importance.head(15)
)


# ============================================================
# 18. SAVE BASELINE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "day14_feature_importance_before.csv",
    index=False
)


# ============================================================
# 19. PLOT BASELINE FEATURE IMPORTANCE
# ============================================================

top_features = feature_importance.head(15)


plt.figure(
    figsize=(10, 6)
)


plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features - Before Feature Removal"
)


plt.tight_layout()

plt.savefig(
    "feature_importance_before.png",
    dpi=300
)

plt.show()


# ============================================================
# 20. DAY 14 CONSTRAINT CHANGE
# ============================================================

print("\n" + "=" * 60)
print("DAY 14 - CONSTRAINT CHANGE")
print("=" * 60)


print(
    f"\nRemoving important feature: "
    f"{FEATURE_TO_REMOVE}"
)


if FEATURE_TO_REMOVE not in X.columns:

    print(
        f"\nERROR: Feature "
        f"'{FEATURE_TO_REMOVE}' "
        "does not exist."
    )

    print("\nAvailable features:")
    print(X.columns.tolist())

    exit()


# ============================================================
# 21. REMOVE IMPORTANT FEATURE
# ============================================================

X_reduced = X.drop(
    columns=[FEATURE_TO_REMOVE]
)


print("\nFeature removed successfully.")


print(
    "\nSales_Amount exists after removal?"
)


print(
    FEATURE_TO_REMOVE in X_reduced.columns
)


print("\nRemaining features:")

for column in X_reduced.columns:
    print(" -", column)


# ============================================================
# 22. TRAIN TEST SPLIT AFTER REMOVAL
# ============================================================

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ============================================================
# 23. IDENTIFY REDUCED FEATURES
# ============================================================

numeric_features_reduced = (
    X_reduced
    .select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    )
    .columns
    .tolist()
)


categorical_features_reduced = (
    X_reduced
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()
)


# ============================================================
# 24. REDUCED DATA PREPROCESSOR
# ============================================================

numeric_transformer_reduced = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer_reduced = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor_reduced = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer_reduced,
            numeric_features_reduced
        ),

        (
            "cat",
            categorical_transformer_reduced,
            categorical_features_reduced
        )
    ]
)


# ============================================================
# 25. ADAPTED MODEL
# ============================================================

adapted_model = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


adapted_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor_reduced
        ),

        (
            "model",
            adapted_model
        )
    ]
)


# ============================================================
# 26. TRAIN ADAPTED MODEL
# ============================================================

print("\nTraining adapted model...")

adapted_pipeline.fit(
    X_train_reduced,
    y_train_reduced
)

print(
    "Adapted model trained successfully."
)


# ============================================================
# 27. ADAPTED MODEL PREDICTIONS
# ============================================================

y_pred_adapted = (
    adapted_pipeline
    .predict(X_test_reduced)
)


# ============================================================
# 28. ADAPTED MODEL METRICS
# ============================================================

mae_adapted = mean_absolute_error(
    y_test_reduced,
    y_pred_adapted
)


mse_adapted = mean_squared_error(
    y_test_reduced,
    y_pred_adapted
)


rmse_adapted = np.sqrt(
    mse_adapted
)


r2_adapted = r2_score(
    y_test_reduced,
    y_pred_adapted
)


print("\n" + "=" * 60)
print("ADAPTED MODEL PERFORMANCE")
print("=" * 60)


print(
    f"\nMAE  : {mae_adapted:.4f}"
)

print(
    f"MSE  : {mse_adapted:.4f}"
)

print(
    f"RMSE : {rmse_adapted:.4f}"
)

print(
    f"R2   : {r2_adapted:.4f}"
)


# ============================================================
# 29. PERFORMANCE COMPARISON
# ============================================================

comparison = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "MSE",
            "RMSE",
            "R2 Score"
        ],

        "Before Feature Removal": [
            mae_baseline,
            mse_baseline,
            rmse_baseline,
            r2_baseline
        ],

        "After Feature Removal": [
            mae_adapted,
            mse_adapted,
            rmse_adapted,
            r2_adapted
        ]
    }
)


print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)

print("\n")

print(comparison.to_string(index=False))


# ============================================================
# 30. CALCULATE PERFORMANCE CHANGES
# ============================================================

mae_change = (
    (
        mae_adapted - mae_baseline
    )
    / abs(mae_baseline)
) * 100


rmse_change = (
    (
        rmse_adapted - rmse_baseline
    )
    / abs(rmse_baseline)
) * 100


r2_change = (
    (
        r2_adapted - r2_baseline
    )
    / abs(r2_baseline)
) * 100


print("\n" + "=" * 60)
print("PERFORMANCE CHANGE")
print("=" * 60)


print(
    f"\nMAE Change  : {mae_change:.2f}%"
)

print(
    f"RMSE Change : {rmse_change:.2f}%"
)

print(
    f"R2 Change   : {r2_change:.2f}%"
)


# ============================================================
# 31. SAVE PERFORMANCE COMPARISON
# ============================================================

comparison.to_csv(
    "day14_performance_comparison.csv",
    index=False
)


# ============================================================
# 32. ADAPTED FEATURE IMPORTANCE
# ============================================================

feature_names_reduced = (
    adapted_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


importances_reduced = (
    adapted_pipeline
    .named_steps["model"]
    .feature_importances_
)


feature_importance_reduced = pd.DataFrame(
    {
        "Feature": feature_names_reduced,
        "Importance": importances_reduced
    }
)


feature_importance_reduced = (
    feature_importance_reduced
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n" + "=" * 60)
print("TOP FEATURES AFTER FEATURE REMOVAL")
print("=" * 60)


print(
    feature_importance_reduced.head(15)
)


# ============================================================
# 33. SAVE ADAPTED FEATURE IMPORTANCE
# ============================================================

feature_importance_reduced.to_csv(
    "day14_feature_importance_after.csv",
    index=False
)


# ============================================================
# 34. PLOT ADAPTED FEATURE IMPORTANCE
# ============================================================

top_features_reduced = (
    feature_importance_reduced.head(15)
)


plt.figure(
    figsize=(10, 6)
)


plt.barh(
    top_features_reduced["Feature"][::-1],
    top_features_reduced["Importance"][::-1]
)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features - After Sales_Amount Removal"
)


plt.tight_layout()


plt.savefig(
    "feature_importance_after.png",
    dpi=300
)


plt.show()


# ============================================================
# 35. ACTUAL VS PREDICTED - BASELINE
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_test,
    y_pred_baseline,
    alpha=0.6
)


plt.xlabel(
    "Actual Total Sales"
)

plt.ylabel(
    "Predicted Total Sales"
)

plt.title(
    "Baseline Model - Actual vs Predicted"
)


plt.tight_layout()


plt.savefig(
    "baseline_actual_vs_predicted.png",
    dpi=300
)


plt.show()


# ============================================================
# 36. ACTUAL VS PREDICTED - ADAPTED
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_test_reduced,
    y_pred_adapted,
    alpha=0.6
)


plt.xlabel(
    "Actual Total Sales"
)

plt.ylabel(
    "Predicted Total Sales"
)

plt.title(
    "Adapted Model - Actual vs Predicted"
)


plt.tight_layout()


plt.savefig(
    "adapted_actual_vs_predicted.png",
    dpi=300
)


plt.show()


# ============================================================
# 37. PERFORMANCE COMPARISON GRAPH
# ============================================================

metrics = [
    "MAE",
    "RMSE",
    "R2"
]


before = [
    mae_baseline,
    rmse_baseline,
    r2_baseline
]


after = [
    mae_adapted,
    rmse_adapted,
    r2_adapted
]


x = np.arange(
    len(metrics)
)


width = 0.35


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    x - width / 2,
    before,
    width,
    label="Before Feature Removal"
)


plt.bar(
    x + width / 2,
    after,
    width,
    label="After Feature Removal"
)


plt.xticks(
    x,
    metrics
)


plt.ylabel(
    "Score"
)


plt.title(
    "Model Performance Before vs After Feature Removal"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "day14_performance_comparison.png",
    dpi=300
)


plt.show()


# ============================================================
# 38. SAVE MODELS
# ============================================================

joblib.dump(
    baseline_pipeline,
    "day14_baseline_model.pkl"
)


joblib.dump(
    adapted_pipeline,
    "day14_adapted_model.pkl"
)


# ============================================================
# 39. FINAL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DAY 14 ANALYSIS")
print("=" * 60)


if r2_adapted < r2_baseline:

    print(
        "\nR2 decreased after removing Sales_Amount."
    )

    print(
        "This indicates that Sales_Amount contained "
        "important predictive information."
    )

elif r2_adapted > r2_baseline:

    print(
        "\nR2 increased after removing Sales_Amount."
    )

    print(
        "The remaining features were sufficient to "
        "produce a stronger model on this test split."
    )

else:

    print(
        "\nR2 remained approximately unchanged."
    )


if rmse_adapted > rmse_baseline:

    print(
        "\nRMSE increased."
    )

    print(
        "Prediction errors became larger after "
        "the feature was removed."
    )

elif rmse_adapted < rmse_baseline:

    print(
        "\nRMSE decreased."
    )

    print(
        "The adapted model produced lower prediction errors."
    )


# ============================================================
# 40. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL DAY 14 SUMMARY")
print("=" * 60)


print(
    f"\nFeature removed: {FEATURE_TO_REMOVE}"
)


print(
    f"Baseline R2: {r2_baseline:.4f}"
)


print(
    f"Adapted R2 : {r2_adapted:.4f}"
)


print(
    f"Baseline RMSE: {rmse_baseline:.4f}"
)


print(
    f"Adapted RMSE : {rmse_adapted:.4f}"
)


print(
    f"\nR2 change: {r2_change:.2f}%"
)


print(
    f"RMSE change: {rmse_change:.2f}%"
)


# ============================================================
# 41. FILES GENERATED
# ============================================================

print("\n" + "=" * 60)
print("FILES GENERATED")
print("=" * 60)


files_generated = [
    "day14_performance_comparison.csv",
    "day14_feature_importance_before.csv",
    "day14_feature_importance_after.csv",
    "day14_baseline_model.pkl",
    "day14_adapted_model.pkl",
    "feature_importance_before.png",
    "feature_importance_after.png",
    "baseline_actual_vs_predicted.png",
    "adapted_actual_vs_predicted.png",
    "day14_performance_comparison.png"
]


for file in files_generated:

    if os.path.exists(file):

        print("✓", file)

    else:

        print("✗", file)


print("\n" + "=" * 60)
print("DAY 14 COMPLETED SUCCESSFULLY!")
print("=" * 60)