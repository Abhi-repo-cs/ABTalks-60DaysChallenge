# Day 24 — Compressing Complex Data with PCA

## Phase
Dimensionality Reduction

## Objective
Apply Principal Component Analysis (PCA) to reduce feature dimensionality while measuring how much variance is retained and how model performance changes.

## Dataset
Breast Cancer Wisconsin dataset from `sklearn.datasets.load_breast_cancer`.

- Samples: 569
- Original features: 30
- Target: binary classification

## Workflow
1. Load the dataset.
2. Split into training and testing data.
3. Standardize the features.
4. Train a Logistic Regression baseline.
5. Apply PCA with 2 and 3 components.
6. Test PCA configurations retaining 90%, 95%, and 99% variance.
7. Compare model performance.
8. Visualize the transformed feature space.
9. Analyze explained and cumulative explained variance.

## Key result
- PC1 + PC2 retain **63.24%** variance.
- PC1 + PC2 + PC3 retain **72.64%** variance.
- 90% variance requires **7** components.
- 95% variance requires **10** components.
- 99% variance requires **17** components.

## Repository structure
```text
Day-24-PCA/
├── PCA.py
├── dataset.csv
├── outputs/
│   ├── pca_2d.png
│   ├── pca_3d.png
│   ├── explained_variance.png
│   ├── cumulative_variance.png
│   └── model_comparison.csv
├── reports/
│   └── variance_analysis.md
└── README.md
```

## Installation
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Run
```bash
python PCA.py
```

## Learning takeaway
PCA creates new orthogonal features called principal components. The goal is not simply to reduce the number of columns, but to find a lower-dimensional representation that preserves as much useful variation as possible.
