# Business Forecasting Report - Day 37

## Objective
Forecast the next 12 months of business revenue from historical monthly revenue data.

## Method
Primary model: **ARIMA(1,1,1)**.
ARIMA is used when `statsmodels` is available. The script includes a deterministic fallback so the workflow remains runnable in minimal environments.

## Forecast Summary
- Forecast horizon: **12 months**
- Average forecast revenue: **22,718.88**
- Last historical revenue: **22,540.00**
- Final forecast revenue: **22,725.57**
- Change from last historical month to final forecast: **0.82%**

## Historical Holdout Accuracy
- MAE: **2,277.92**
- RMSE: **2,553.36**
- MAPE: **10.73%**

Lower MAE/RMSE/MAPE indicates better predictive accuracy on the holdout period.

## Business Implications
- Revenue forecasts support budgeting, staffing, inventory/capacity planning, and investment decisions.
- Forecast uncertainty should be considered before committing large resources.
- Forecasts should be refreshed as new monthly revenue becomes available.
- Model accuracy should be monitored using holdout or rolling-origin evaluation rather than relying only on visual fit.

## Limitations
- This demonstration uses historical revenue only and does not include promotions, pricing, macroeconomic variables, or customer-level drivers.
- A single ARIMA specification may not be optimal for every business; model selection and backtesting are recommended for production use.