# Day 48 — Pipeline Architecture Report

## 1. Objective
Design a reusable, leakage-resistant preprocessing workflow for the Customer Intelligence Platform.

## 2. Architecture

```text
Raw Customer Data
       |
       v
Schema + Data Quality Checks
       |
       v
Duplicate Removal
       |
       v
Feature Engineering
       |
       +----------------------+
       |                      |
       v                      v
Numerical Features      Categorical Features
       |                      |
       v                      v
Median Imputation       Most-Frequent Imputation
       |                      |
       v                      v
Standard Scaling        One-Hot Encoding
       |                      |
       +----------+-----------+
                  |
                  v
          Processed Features
                  |
                  v
             ML Model
```

## 3. Preprocessing Decisions

| Area | Decision | Rationale |
|---|---|---|
| Duplicates | Exact duplicates removed | Prevent repeated observations |
| Numeric missing values | Median imputation | Robust to skew/outliers |
| Categorical missing values | Most-frequent imputation | Simple stable baseline |
| Numeric transformation | StandardScaler | Comparable feature scale |
| Categorical encoding | OneHotEncoder | ML-compatible representation |
| Unknown categories | `handle_unknown="ignore"` | Prevent inference failures |
| Reusability | Pipeline + ColumnTransformer | Same transformations during training/inference |
| Leakage control | Fit on training split only | Prevent test-set information leakage |

## 4. Feature Engineering

The pipeline creates:
- `avg_order_value`
- `orders_per_month`
- `spend_per_month`

These features translate raw transactional values into customer behavior indicators.

## 5. Leakage Prevention

The transformer is fitted using `X_train` only:

```python
preprocessor.fit(X_train)
X_test_processed = preprocessor.transform(X_test)
```

The test set is never used to estimate imputation statistics, scaling parameters, or category mappings.

## 6. Outputs

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `models/preprocessor.joblib`

## 7. Production Considerations

For a production deployment, the next iterations should add:
- Schema validation
- Data drift monitoring
- Pipeline versioning
- Automated tests
- Dataset lineage
- Model/data quality monitoring
- Orchestration with Airflow, Prefect, or similar tooling

## 8. Conclusion

The Day 48 pipeline converts raw customer records into consistent, machine-learning-ready features while keeping preprocessing reusable and reducing the risk of data leakage.
