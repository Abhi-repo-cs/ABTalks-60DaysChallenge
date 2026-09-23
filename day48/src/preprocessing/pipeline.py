from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic customer-level features."""
    data = df.copy()

    # Avoid division by zero and preserve missing values until imputation.
    data["avg_order_value"] = data["total_spend"] / data["orders"].replace(0, pd.NA)
    data["orders_per_month"] = data["orders"] / data["tenure_months"].replace(0, pd.NA)
    data["spend_per_month"] = data["total_spend"] / data["tenure_months"].replace(0, pd.NA)

    return data


def build_preprocessor(X: pd.DataFrame):
    """Build a reusable ColumnTransformer from the input schema."""
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numerical_features, categorical_features
