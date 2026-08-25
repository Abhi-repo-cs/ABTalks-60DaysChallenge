# Day 23 - Feature Selection Report

## Objective
Identify important predictive signals and remove weak or redundant features.

## Dataset
- Rows: 1200
- Original features: 12
- Target: `churn`

## Methods
1. Correlation analysis
2. Random Forest feature importance
3. Removal of highly correlated features (> 0.90)
4. Selection of features with at least median Random Forest importance
5. Before/after model comparison

## Selected Features
- tenure_months
- purchase_frequency
- avg_order_value
- discount_used
- loyalty_score
- noise_feature

## Correlated Features Removed
- income_thousands
- sessions_duplicate

## Performance

| Model | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| All Features | 12 | 0.7792 | 0.6842 | 0.3881 | 0.4952 | 0.7665 |
| Selected Features | 6 | 0.7750 | 0.6444 | 0.4328 | 0.5179 | 0.7371 |

## Conclusion
Feature selection reduced the number of inputs from 12 to 6.
The selected model should be preferred if it maintains comparable performance because it
is simpler, easier to interpret, and uses fewer input variables.
