# Day 49 — Baseline Capstone Prototype Analysis

## Objective
Integrate the Day 48 preprocessing layer with a first predictive module and generate measurable baseline outputs.

## Baseline Architecture

```text
Raw Customer Data
       ↓
Cleaning + Feature Engineering
       ↓
Train/Test Split
       ↓
Reusable Preprocessor
       ↓
Logistic Regression
       ↓
Churn Probability
       ↓
Prediction Outputs + Evaluation Metrics
```

## Baseline Model
**Logistic Regression** was selected as the initial classifier because it is fast, interpretable, and useful as a benchmark before introducing more complex algorithms.

## Integrated Features
- Average order value
- Orders per month
- Spend per month
- Age
- Income
- Tenure
- Orders
- Total spend
- Complaints
- Preferred channel
- City

## Strengths
1. End-to-end workflow is executable.
2. Preprocessing and modeling are connected through a single sklearn Pipeline.
3. Missing values and categorical variables are handled consistently.
4. Test data is not used to fit preprocessing transformations.
5. Prediction probabilities are generated for business prioritization.
6. Metrics provide a baseline for future model comparison.

## Weaknesses / Limitations
1. The included dataset is a small demonstration dataset.
2. Logistic Regression may not capture nonlinear customer behavior.
3. The dataset is not longitudinal enough for production churn modeling.
4. No production data-drift monitoring is included yet.
5. Threshold selection has not been optimized for business costs.
6. Model performance should not be generalized from the sample dataset to real customers.

## Next Iterations
- Compare Random Forest and Gradient Boosting.
- Add cross-validation.
- Tune the decision threshold using business cost.
- Add model explainability.
- Introduce temporal validation for real churn data.
- Add data-quality tests and drift monitoring.
- Integrate segmentation and forecasting modules.
- Expose predictions through an API/dashboard.

## Definition of Done
The Day 49 prototype successfully demonstrates the complete path:

**Data → Preprocessing → Model → Prediction → Evaluation → Saved Artifact**
