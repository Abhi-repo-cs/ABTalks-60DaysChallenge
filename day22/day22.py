"""
Day 22 - Feature Engineering: Categorical Feature Encoding

This script:
1. Loads a CSV dataset
2. Identifies categorical columns
3. Applies Label Encoding
4. Applies One-Hot Encoding
5. Compares dataset structure before/after encoding
6. Trains a baseline Random Forest model
7. Compares model performance
8. Saves encoded datasets and performance results

Usage:
    python day22.py

Default dataset:
    data/raw_dataset.csv

If your dataset/target column has a different name, edit DATA_PATH and TARGET_COLUMN below.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw_dataset.csv"

# Set this to your target column.
# If None, the script will use the last column in the dataset.
TARGET_COLUMN = None

TEST_SIZE = 0.20
RANDOM_STATE = 42

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    path = Path(DATA_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Place your CSV file there or change DATA_PATH in day22.py."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    print("\n" + "=" * 70)
    print("DAY 22 - FEATURE ENCODING")
    print("=" * 70)

    print(f"Dataset: {path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# DATASET INFORMATION
# ============================================================

def inspect_dataset(df):
    print("\n" + "-" * 70)
    print("1. ORIGINAL DATASET")
    print("-" * 70)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nCategorical columns:")
    if categorical_cols:
        for column in categorical_cols:
            print(f"  - {column}")
    else:
        print("  No categorical columns found.")

    print(f"\nOriginal shape: {df.shape}")

    return categorical_cols


# ============================================================
# PREPARE TARGET
# ============================================================

def prepare_target(df, target_column):
    if target_column is None:
        target_column = df.columns[-1]
        print(
            f"\nTARGET_COLUMN was not specified. "
            f"Using last column: '{target_column}'"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # Handle missing target values.
    valid_rows = y.notna()
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    # Classification target encoding if necessary.
    target_encoder = None

    if y.dtype == "object" or str(y.dtype) == "category" or y.dtype == "bool":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

        print("\nTarget encoding:")
        for original, encoded in zip(
            target_encoder.classes_,
            range(len(target_encoder.classes_))
        ):
            print(f"  {original} -> {encoded}")

    return X, y, target_column


# ============================================================
# COMMON FEATURE CLEANING
# ============================================================

def prepare_features(X):
    X = X.copy()

    # Convert boolean columns to strings so they can be handled
    # consistently as categorical features.
    bool_cols = X.select_dtypes(include=["bool"]).columns

    for column in bool_cols:
        X[column] = X[column].astype(str)

    # Fill missing categorical values with the most frequent value.
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for column in categorical_cols:
        if X[column].isna().any():
            mode = X[column].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            X[column] = X[column].fillna(fill_value)

    # Fill missing numerical values with median.
    numerical_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for column in numerical_cols:
        if X[column].isna().any():
            X[column] = X[column].fillna(X[column].median())

    return X


# ============================================================
# LABEL ENCODING
# ============================================================

def apply_label_encoding(X):
    X_encoded = X.copy()
    encoders = {}

    categorical_cols = X_encoded.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for column in categorical_cols:
        encoder = LabelEncoder()
        X_encoded[column] = encoder.fit_transform(
            X_encoded[column].astype(str)
        )
        encoders[column] = encoder

    return X_encoded, encoders


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def apply_one_hot_encoding(X):
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = [
        column for column in X.columns
        if column not in categorical_cols
    ]

    if categorical_cols:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            # Compatibility with older scikit-learn versions.
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        transformer = ColumnTransformer(
            transformers=[
                ("categorical", encoder, categorical_cols)
            ],
            remainder="passthrough"
        )

        encoded_array = transformer.fit_transform(X)

        encoded_categorical_names = (
            transformer.named_transformers_["categorical"]
            .get_feature_names_out(categorical_cols)
            .tolist()
        )

        feature_names = encoded_categorical_names + numerical_cols

        X_encoded = pd.DataFrame(
            encoded_array,
            columns=feature_names,
            index=X.index
        )

    else:
        X_encoded = X.copy()

    return X_encoded


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(X, y, encoding_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = {
        "Encoding": encoding_name,
        "Original_Features": X.shape[1],
        "Training_Rows": X_train.shape[0],
        "Testing_Rows": X_test.shape[0],
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    # Load
    df = load_dataset()

    # Inspect
    categorical_cols = inspect_dataset(df)

    # Target
    X, y, target_column = prepare_target(
        df,
        TARGET_COLUMN
    )

    # Feature preparation
    X = prepare_features(X)

    print("\n" + "-" * 70)
    print("2. FEATURE INFORMATION")
    print("-" * 70)

    print(f"Target column: {target_column}")
    print(f"Feature columns before encoding: {X.shape[1]}")

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")

    # --------------------------------------------------------
    # Label Encoding
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("3. LABEL ENCODING")
    print("-" * 70)

    X_label, label_encoders = apply_label_encoding(X)

    print(f"Shape after Label Encoding: {X_label.shape}")

    print("\nEncoded categorical columns:")
    if label_encoders:
        for column, encoder in label_encoders.items():
            mapping = {
                class_name: int(index)
                for index, class_name in enumerate(encoder.classes_)
            }
            print(f"  {column}: {mapping}")
    else:
        print("  No categorical columns to encode.")

    label_path = OUTPUT_DIR / "label_encoded.csv"

    label_output = X_label.copy()
    label_output[target_column] = y
    label_output.to_csv(label_path, index=False)

    # --------------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. ONE-HOT ENCODING")
    print("-" * 70)

    X_onehot = apply_one_hot_encoding(X)

    print(f"Shape after One-Hot Encoding: {X_onehot.shape}")

    onehot_path = OUTPUT_DIR / "onehot_encoded.csv"

    onehot_output = X_onehot.copy()
    onehot_output[target_column] = y
    onehot_output.to_csv(onehot_path, index=False)

    # --------------------------------------------------------
    # Save dataset comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "Version": [
            "Original",
            "Label Encoding",
            "One-Hot Encoding"
        ],
        "Rows": [
            X.shape[0],
            X_label.shape[0],
            X_onehot.shape[0]
        ],
        "Features": [
            X.shape[1],
            X_label.shape[1],
            X_onehot.shape[1]
        ]
    })

    comparison_path = Path("dataset_structure_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "-" * 70)
    print("5. DATASET STRUCTURE COMPARISON")
    print("-" * 70)
    print(comparison.to_string(index=False))

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("6. BASELINE MODEL PERFORMANCE")
    print("-" * 70)

    print("\nModel: Random Forest Classifier")
    print(f"Test size: {TEST_SIZE}")
    print(f"Random state: {RANDOM_STATE}")

    results = []

    try:
        label_results = evaluate_model(
            X_label,
            y,
            "Label Encoding"
        )
        results.append(label_results)
    except ValueError as error:
        print(f"Label Encoding model could not be trained: {error}")

    try:
        onehot_results = evaluate_model(
            X_onehot,
            y,
            "One-Hot Encoding"
        )
        results.append(onehot_results)
    except ValueError as error:
        print(f"One-Hot Encoding model could not be trained: {error}")

    if not results:
        raise RuntimeError(
            "No model could be trained. Check your target column and dataset."
        )

    results_df = pd.DataFrame(results)

    performance_path = Path("performance_comparison.csv")
    results_df.to_csv(performance_path, index=False)

    print("\nPerformance comparison:")
    print(
        results_df[
            [
                "Encoding",
                "Original_Features",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Final analysis
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("7. ANALYSIS")
    print("-" * 70)

    if len(results_df) >= 2:
        best_accuracy_row = results_df.loc[
            results_df["Accuracy"].idxmax()
        ]

        best_f1_row = results_df.loc[
            results_df["F1_Score"].idxmax()
        ]

        accuracy_difference = (
            results_df.iloc[0]["Accuracy"]
            - results_df.iloc[1]["Accuracy"]
        )

        print(
            f"Best Accuracy: {best_accuracy_row['Encoding']} "
            f"({best_accuracy_row['Accuracy']:.4f})"
        )

        print(
            f"Best F1 Score: {best_f1_row['Encoding']} "
            f"({best_f1_row['F1_Score']:.4f})"
        )

        print(
            f"Accuracy difference between the first two methods: "
            f"{abs(accuracy_difference):.4f}"
        )

        feature_difference = (
            X_onehot.shape[1] - X_label.shape[1]
        )

        print(
            f"One-Hot Encoding created {feature_difference:+d} "
            f"features compared with Label Encoding."
        )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print(f"1. {label_path}")
    print(f"2. {onehot_path}")
    print(f"3. {comparison_path}")
    print(f"4. {performance_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
