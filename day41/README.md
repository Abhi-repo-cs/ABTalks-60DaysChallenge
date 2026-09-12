# Day 41 — Predictive Business Analytics

## Project
**Predictive Customer Risk System**

This project combines customer behavior, segmentation, and churn signals to predict
customer churn risk, rank customers, visualize high-risk groups, and recommend
retention strategies.

## Objectives
- Combine behavioral and segmentation features.
- Build a predictive churn-risk model.
- Rank customers by churn probability.
- Visualize high-risk groups.
- Recommend targeted retention strategies.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## How to Run

```bash
pip install -r requirements.txt
python day41_risk_prediction.py
```

The script creates:
- `outputs/customer_risk_scores.csv`
- `outputs/risk_summary.csv`
- `outputs/risk_dashboard.png`
- `outputs/business_recommendation_report.md`

## Risk Bands
- **High:** churn probability >= 70%
- **Medium:** churn probability 40–69.9%
- **Low:** churn probability < 40%

## Business Value
A predictive risk system helps a business move from reactive churn handling to
proactive retention. Instead of treating every customer equally, teams can prioritize
customers based on estimated risk and the underlying behavioral signal.

## Important Note
The included dataset is synthetic and is intended for learning/demo purposes.
The model should be validated on real historical customer data before business use.
