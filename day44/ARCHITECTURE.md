# Dashboard Architecture Explanation

1. **Data layer** — accepts the included synthetic CSV or a user-uploaded CSV.
2. **Processing layer** — pandas validates required fields, applies filters and calculates KPIs.
3. **Analytics layer** — aggregates customers/revenue by segment and calculates churn and churn-risk metrics.
4. **Visualization layer** — Plotly renders segmentation, churn and risk distributions.
5. **Application layer** — Streamlit provides the interactive business UI and watchlist.
