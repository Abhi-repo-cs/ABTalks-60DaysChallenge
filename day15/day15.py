# ============================================================
# DAY 15 - DATA SCIENCE CHALLENGE
# Classification Foundations
#
# Project:
# Predicting New vs Returning Customers
# using Logistic Regression
#
# Dataset:
# Sales Dataset
#
# Target:
# Customer_Type
#
# Classes:
# New
# Returning
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

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_FILE = "sales_data_cleaned.csv"

TARGET = "Customer_Type"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# 3. START
# ============================================================

print("=" * 70)
print("DAY 15 - CLASSIFICATION FOUNDATIONS")
print("Predicting New vs Returning Customers")
print("=" * 70)


# ============================================================
# 4. CHECK DATASET
# ============================================================

if not os.path.exists(DATA_FILE):

    print("\nERROR: Dataset not found.")

    print(
        f"\nLooking for: {DATA_FILE}"
    )

    print("\nFiles in current folder:")

    for file in os.listdir("."):
        print(" -", file)

    print(
        "\nChange DATA_FILE at the top of the program "
        "to your actual CSV filename."
    )

    exit()


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 6. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nColumns:")

for column in df.columns:
    print(" -", column)


print("\nData types:")

print(df.dtypes)


print("\nMissing values:")

print(df.isnull().sum())


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

if duplicates > 0:

    df = df.drop_duplicates()

    print(
        "Duplicates removed."
    )


# ============================================================
# 8. CHECK TARGET
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
# 9. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    df[TARGET].value_counts()
)


print("\nTarget percentage:")

print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 10. DATE PROCESSING
# ============================================================

if "Sale_Date" in df.columns:

    print("\nProcessing Sale_Date...")

    df["Sale_Date"] = pd.to_datetime(
        df["Sale_Date"],
        errors="coerce"
    )

    # Create useful date features
    df["Sale_Year"] = (
        df["Sale_Date"].dt.year
    )

    df["Sale_Month_Num"] = (
        df["Sale_Date"].dt.month
    )

    df["Sale_Day"] = (
        df["Sale_Date"].dt.day
    )

    df["Sale_DayOfWeek"] = (
        df["Sale_Date"].dt.dayofweek
    )

    # Remove original date
    df = df.drop(
        columns=["Sale_Date"]
    )

    print(
        "Date features created."
    )


# ============================================================
# 11. CREATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


print("\n" + "=" * 70)
print("FEATURES AND TARGET")
print("=" * 70)

print("\nTarget:")
print(TARGET)

print("\nFeatures:")

for column in X.columns:
    print(" -", column)


# ============================================================
# 12. REMOVE IDENTIFIER COLUMNS
# ============================================================

# Product_ID is an identifier rather than a meaningful
# predictive feature.

columns_to_drop = []

if "Product_ID" in X.columns:

    columns_to_drop.append(
        "Product_ID"
    )


# Region_and_Sales_Rep duplicates information already present
# in Region and Sales_Rep, so remove it to avoid redundancy.

if "Region_and_Sales_Rep" in X.columns:

    columns_to_drop.append(
        "Region_and_Sales_Rep"
    )


if len(columns_to_drop) > 0:

    X = X.drop(
        columns=columns_to_drop
    )

    print(
        "\nRemoved identifier/redundant columns:"
    )

    for column in columns_to_drop:
        print(" -", column)


# ============================================================
# 13. IDENTIFY NUMERICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()


# ============================================================
# 14. IDENTIFY CATEGORICAL FEATURES
# ============================================================

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
# 15. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 16. NUMERICAL PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]
)


# ============================================================
# 17. CATEGORICAL PREPROCESSING
# ============================================================

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


# ============================================================
# 18. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )

    ]
)


# ============================================================
# 19. LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)


# ============================================================
# 20. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ]
)


# ============================================================
# 21. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print(
    "Model trained successfully."
)


# ============================================================
# 22. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# 23. PREDICTION PROBABILITIES
# ============================================================

y_probability = pipeline.predict_proba(
    X_test
)


# ============================================================
# 24. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

classes = pipeline.classes_


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nClasses:")

print(classes)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 27. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.colorbar()


tick_marks = np.arange(
    len(classes)
)


plt.xticks(
    tick_marks,
    classes,
    rotation=45
)

plt.yticks(
    tick_marks,
    classes
)


# Add values inside cells

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


plt.tight_layout()


plt.savefig(
    "day15_confusion_matrix.png",
    dpi=300
)


plt.show()


# ============================================================
# 28. PREDICTION RESULTS DATAFRAME
# ============================================================

results = X_test.copy()

results["Actual"] = y_test.values

results["Predicted"] = y_pred


# Prediction confidence

results["Prediction_Confidence"] = (
    y_probability.max(axis=1)
)


print("\n" + "=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

print(
    results.head(20)
)


# ============================================================
# 29. IDENTIFY INCORRECT PREDICTIONS
# ============================================================

incorrect = results[
    results["Actual"]
    !=
    results["Predicted"]
]


print("\nNumber of incorrect predictions:")

print(
    len(incorrect)
)


print("\nIncorrect predictions:")

print(
    incorrect.head(20)
)


# ============================================================
# 30. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)


print(
    "\nIMPORTANT:"
)

print(
    "Because this dataset predicts New vs Returning "
    "rather than actual Churn, FP/FN are interpreted "
    "relative to the class ordering."
)


print(
    "\nClass ordering used by sklearn:"
)

print(classes)


# For binary classification

if len(classes) == 2:

    negative_class = classes[0]

    positive_class = classes[1]

    tn, fp, fn, tp = cm.ravel()

    print(
        f"\nNegative class: {negative_class}"
    )

    print(
        f"Positive class: {positive_class}"
    )

    print(
        f"\nTrue Negatives : {tn}"
    )

    print(
        f"False Positives: {fp}"
    )

    print(
        f"False Negatives: {fn}"
    )

    print(
        f"True Positives : {tp}"
    )


# ============================================================
# 31. BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS IMPLICATIONS")
print("=" * 70)


if len(classes) == 2:

    print(
        f"""
False Positive:
The model predicts '{positive_class}'
but the actual class is '{negative_class}'.

Business impact:
The company may treat a customer as belonging
to the positive segment when that assumption is
incorrect. This can result in unnecessary
targeted campaigns or incorrect customer
segmentation.

False Negative:
The model predicts '{negative_class}'
but the actual class is '{positive_class}'.

Business impact:
A genuinely positive customer may be missed.
The company could lose an opportunity to
target or retain that customer appropriately.
"""
    )


# ============================================================
# 32. SAVE PREDICTION RESULTS
# ============================================================

results.to_csv(
    "day15_prediction_results.csv",
    index=False
)


print(
    "\nPrediction results saved as:"
)

print(
    "day15_prediction_results.csv"
)


# ============================================================
# 33. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]

})


metrics.to_csv(
    "day15_model_metrics.csv",
    index=False
)


# ============================================================
# 34. FEATURE COEFFICIENTS
# ============================================================

try:

    feature_names = (
        pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )


    coefficients = (
        pipeline
        .named_steps["model"]
        .coef_[0]
    )


    coefficient_df = pd.DataFrame({

        "Feature": feature_names,

        "Coefficient": coefficients,

        "Absolute_Coefficient": np.abs(
            coefficients
        )

    })


    coefficient_df = (
        coefficient_df
        .sort_values(
            by="Absolute_Coefficient",
            ascending=False
        )
    )


    print("\n" + "=" * 70)
    print("IMPORTANT LOGISTIC REGRESSION FEATURES")
    print("=" * 70)


    print(
        coefficient_df.head(15)
    )


    coefficient_df.to_csv(
        "day15_feature_coefficients.csv",
        index=False
    )


except Exception as e:

    print(
        "\nCould not calculate coefficients:"
    )

    print(e)


# ============================================================
# 35. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 15 SUMMARY")
print("=" * 70)


print(
    f"\nDataset size: {df.shape}"
)


print(
    f"Target: {TARGET}"
)


print(
    f"Model: Logistic Regression"
)


print(
    f"\nAccuracy: {accuracy:.4f}"
)


print(
    f"Precision: {precision:.4f}"
)


print(
    f"Recall: {recall:.4f}"
)


print(
    f"F1 Score: {f1:.4f}"
)


print(
    f"\nIncorrect predictions: "
    f"{len(incorrect)}"
)


print(
    "\nFiles generated:"
)


files = [

    "day15_confusion_matrix.png",

    "day15_prediction_results.csv",

    "day15_model_metrics.csv",

    "day15_feature_coefficients.csv"

]


for file in files:

    if os.path.exists(file):

        print("✓", file)

    else:

        print("✗", file)


print("\n" + "=" * 70)

print(
    "DAY 15 COMPLETED SUCCESSFULLY!"
)

print("=" * 70)