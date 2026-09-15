# Customer Intelligence Dashboard

A Streamlit business intelligence dashboard for customer analytics, segmentation and churn-risk monitoring.

## Features
- KPI cards for customers, monthly revenue, churn rate and high-risk customers
- Interactive segment and contract filters
- Customer segmentation and churn analytics
- Churn-risk distribution
- High-risk customer watchlist
- CSV upload for replacing demo data
- Business-oriented interpretation notes

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture
CSV / uploaded data → pandas validation & filtering → KPI aggregation + churn-risk analysis → Plotly visualizations → Streamlit UI.

## Data
`data/customer_data.csv` is synthetic demo data generated for this project.

## Screenshots
Add screenshots of the running dashboard here before submission.

## LinkedIn reflection
See `LINKEDIN_REFLECTION.md`.
