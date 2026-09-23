# Day 50 — Model Comparison & Optimization Report

## Objective
Improve the Day 49 baseline by comparing multiple ML algorithms and tuning their hyperparameters using cross-validation.

## Models

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Nonlinear ensemble benchmark |
| Gradient Boosting | Sequential boosting benchmark |

## Optimization Method
- Stratified 5-fold cross-validation
- GridSearchCV
- ROC-AUC as the primary tuning metric
- Accuracy, precision, recall and F1 as supporting metrics
- Train vs test ROC-AUC gap for generalization analysis

## Engineering Tradeoffs

### Logistic Regression
**Advantages:** fast, interpretable, simple to deploy.

**Limitations:** linear decision boundary may miss nonlinear customer behavior.

### Random Forest
**Advantages:** captures nonlinear relationships and feature interactions; relatively robust.

**Limitations:** larger models can consume more memory and may be less interpretable.

### Gradient Boosting
**Advantages:** strong performance on structured/tabular data and flexible nonlinear modeling.

**Limitations:** more sensitive to hyperparameters and can overfit.

## Overfitting / Underfitting Analysis

The `generalization_gap` column is calculated as:

`train ROC-AUC - test ROC-AUC`

A large positive gap indicates potential overfitting. Low train and test performance can indicate underfitting or weak feature representation.

## Important Evaluation Principle

The test set is held out until final evaluation. Hyperparameters are selected using cross-validation on the training set. This prevents the test set from becoming part of model selection.

## Production Caveats

The included dataset is a demonstration dataset. Its results should not be presented as evidence of real customer behavior or production model quality.

Before deployment, the system should use:
- Real longitudinal customer data
- Time-aware validation
- Class-imbalance analysis
- Probability calibration
- Business-cost-based threshold selection
- Explainability
- Data drift monitoring
- Model monitoring and retraining policies

## Day 50 Outcome

The capstone now has a repeatable optimization workflow that can establish a baseline, compare candidate algorithms, tune hyperparameters, quantify generalization, and save an optimized model artifact.
