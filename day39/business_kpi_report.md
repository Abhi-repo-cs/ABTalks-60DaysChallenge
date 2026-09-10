# Business KPI Report — Day 39

## 1. Objective
This project builds a KPI monitoring system that tracks customer intelligence, revenue performance, retention, and unit-economics indicators.

## 2. KPI Framework
| KPI | Definition | Executive use |
|---|---|---|
| Total Customers | Current customer base | Measures scale |
| Customer Growth | MoM change in customers | Tracks acquisition momentum |
| Revenue | Active customers × ARPU | Measures financial performance |
| Revenue Growth | MoM change in revenue | Tracks growth acceleration |
| Retention Rate | 1 − churn rate | Measures customer stickiness |
| ARPU | Revenue / active customers | Measures monetization |
| AOV | Revenue / orders | Measures transaction value |
| Net Customer Adds | New customers − churned customers | Measures base expansion |

## 3. Executive Findings
Using the latest month in the supplied analytical dataset (August 2026):

- Customer base: **11,033**
- Revenue: **$827,452**
- Retention: **96.2%**
- ARPU: **$77.92**
- Average order value: **$94.91**
- Month-over-month revenue change: **-2.2%**
- Month-over-month customer change: **-4.0%**

## 4. Business Interpretation
The dashboard is designed around three executive questions:

1. **Are we growing?** — customer and revenue trends.
2. **Are customers staying?** — retention and churn.
3. **Are customers becoming more valuable?** — ARPU and AOV.

A healthy KPI pattern is sustained revenue and customer growth combined with stable or improving retention. If revenue grows while retention deteriorates, management should investigate whether acquisition is masking customer-quality problems.

## 5. Dashboard Design
The Streamlit app provides:
- Interactive date filtering
- KPI cards with month-over-month deltas
- Revenue trend visualization
- Customer and active-customer trends
- Retention trend
- Acquisition versus churn comparison
- Executive KPI summary
- Downloadable/raw KPI table through the app's data view

## 6. Recommendations
- Monitor retention as a leading indicator of recurring-revenue risk.
- Compare acquisition growth with churn to distinguish healthy expansion from customer replacement.
- Track ARPU and AOV alongside revenue so executives can separate volume growth from monetization growth.
- Establish KPI thresholds and alerts for material month-over-month deterioration.
- Connect the dashboard to a production database or BI pipeline for real-time monitoring.

## 7. Limitations
The included dataset is a synthetic demonstration dataset intended for learning and portfolio use. In production, KPI definitions should be aligned with the company's finance, CRM, and customer-success systems, with consistent cohort and revenue-recognition rules.
