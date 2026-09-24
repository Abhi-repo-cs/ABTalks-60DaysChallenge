# Day 53 — Real-Time Analytics Workflow

## Objective
Enable a user to submit live customer attributes and receive an immediate churn-risk prediction that is reflected in the analytics dashboard.

## Workflow
1. **Collect live input** — behavior, engagement, service and customer attributes.
2. **Validate & transform** — validate numeric fields and use controlled categorical values; production should reuse the training preprocessing pipeline.
3. **Predict** — return churn probability, risk band and decision signals.
4. **Update dashboard** — refresh KPIs and append a prediction event.
5. **Act** — route high-risk customers to retention workflows and monitor medium-risk customers.

## Architecture
```text
React Form
   ↓
REST API / FastAPI
   ↓
Input Validation + Preprocessing
   ↓
Optimized ML Model
   ↓
Prediction Probability
   ↓
Risk Band + Explanation
   ↓
Dashboard / Event Stream
```

## Production note
This package uses a transparent local scoring function so the complete workflow runs without a backend. Replace it with the optimized model-serving API from the earlier capstone stages for production.
