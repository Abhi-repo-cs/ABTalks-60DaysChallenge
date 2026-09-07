# Day 36 - Time Series Analytics: Customer Growth Forecasting

## Objective
Forecast future customer growth for the next 30 days from historical daily data.

## Dataset
A synthetic e-commerce dataset is used instead of an external dataset. It contains
365 daily records with growth, weekly seasonality, monthly variation, noise, and
promotional spikes.

## Workflow
Historical Data → Trend Visualization → Seasonality → Holt-Winters Model →
30-Day Forecast → Business Interpretation

## Model
**Holt-Winters Exponential Smoothing**
- Additive trend
- Additive seasonality
- 7-day seasonal period

## Run
```bash
pip install -r requirements.txt
python data.py
python day36.py
```

## Outputs
- `customer_growth.csv` — historical dataset
- `trend_visualization.png` — trend and moving averages
- `forecast_30_days.png` — historical data + 30-day prediction
- `forecast_output.csv` — prediction values
- `forecasting_observations.md` — observations and risks

## Real-World Impact
Forecasting can support marketing planning, staffing, resource allocation,
capacity planning, and customer-acquisition targets.

## Limitations
Synthetic data is for learning. Unexpected promotions, holidays, competition,
pricing changes, economic conditions, and product events can reduce forecast
accuracy.

## Future Improvements
Add external variables, compare ARIMA/Prophet/ML models, and evaluate forecasts
with MAE and RMSE when actual future data becomes available.
