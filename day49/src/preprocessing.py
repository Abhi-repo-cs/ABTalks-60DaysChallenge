from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop_duplicates()
    data["avg_order_value"] = data["total_spend"] / data["orders"].replace(0, pd.NA)
    data["orders_per_month"] = data["orders"] / data["tenure_months"].replace(0, pd.NA)
    data["spend_per_month"] = data["total_spend"] / data["tenure_months"].replace(0, pd.NA)
    return data


def build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", num_pipe, numeric),
        ("categorical", cat_pipe, categorical),
    ])

    return preprocessor, numeric, categorical
