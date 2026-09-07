# Forecasting Observations and Risks

## Model
Holt-Winters Exponential Smoothing with additive trend and 7-day seasonality.

## Observations
- Historical period: 2025-09-01 to 2026-08-31
- First 30-day average: 46.9 customers/day
- Latest 30-day average: 68.3 customers/day
- Early-to-recent change: 45.6%
- Strongest weekday: Sunday
- Weakest weekday: Monday
- Predicted customers for next 30 days: 2140
- Average forecast: 71.3 customers/day

## Business Impact
Forecasting supports marketing planning, customer-support staffing,
capacity allocation, campaign timing, and acquisition targets.

## Risks
- The dataset is synthetic and intended for learning.
- Forecasts assume historical patterns continue.
- Promotions, holidays, competitors, pricing, product launches, and economic
  conditions can change actual growth.
- Forecasting estimates future behavior but does not prove causation.

## Next Steps
Add marketing spend, campaigns, holidays, and promotions. Compare Holt-Winters
with ARIMA, Prophet, and machine-learning models. Evaluate predictions using
MAE and RMSE after actual future data becomes available.
