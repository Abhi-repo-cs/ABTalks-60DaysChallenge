# Day 24 — PCA Variance Analysis

## Dataset
The experiment uses the Breast Cancer Wisconsin dataset from scikit-learn.
It contains 569 observations and 30 original numerical features.

## Why scaling was used
PCA is based on variance. Features with larger numerical scales can dominate the principal components, so the features were standardized using `StandardScaler` before PCA.

## Explained variance
The first two principal components retain **63.24%** of the total variance.
The first three components retain **72.64%**.

The number of components required to retain:
- 90% variance: **7**
- 95% variance: **10**
- 99% variance: **17**

## Model comparison
The baseline Logistic Regression model uses all 30 features.
PCA models were evaluated using the same train/test split and Logistic Regression classifier.

| Experiment | Components | Explained Variance | Accuracy | F1 Score |
|---|---:|---:|---:|---:|
| Baseline (no PCA) | 30 | — | 0.9825 | 0.9861 |
| 2 components | 2 | 63.36% | 0.9474 | 0.9577 |
| 3 components | 3 | 72.90% | 0.9211 | 0.9362 |
| 90% variance | 7 | 91.26% | 0.9474 | 0.9577 |
| 95% variance | 10 | 95.27% | 0.9737 | 0.9790 |
| 99% variance | 17 | 99.15% | 0.9649 | 0.9718 |

## Interpretation
PCA successfully transforms the original feature space into a smaller set of orthogonal principal components. However, reducing the dataset to only 2 or 3 components may discard information that is useful for prediction. The variance-retention experiments show how many components are needed when a higher information-retention target is required.

The best accuracy among the tested configurations was **0.9825** using **Baseline (no PCA)**.

## Conclusion
PCA is useful when dimensionality, redundancy, visualization, or computational cost is a concern. The appropriate number of components should be selected based on the required balance between information retention, model performance, interpretability, and computational efficiency.
