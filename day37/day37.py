"""
Day 37 - Business Forecasting
Revenue forecasting using ARIMA, with a lightweight fallback model.

Outputs:
- revenue_data.csv (generated if missing)
- forecast_visualization.png
- business_forecasting_report.md
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("revenue_data.csv")
FORECAST_FILE = Path("revenue_forecast.csv")
PLOT_FILE = Path("forecast_visualization.png")
REPORT_FILE = Path("business_forecasting_report.md")
FORECAST_PERIODS = 12


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "revenue_data.csv not found. Place a CSV with date and revenue columns "
            "in the same folder and rerun."
        )
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["date", "revenue"]).sort_values("date")
    return df.set_index("date").asfreq("MS")


def fit_forecast(series):
    """Use ARIMA when statsmodels is installed; otherwise use seasonal-naive fallback."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series, order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(FORECAST_PERIODS)
        method = "ARIMA(1,1,1)"
        return forecast, method
    except Exception as exc:
        print(f"ARIMA unavailable ({exc}); using seasonal-naive fallback.")
        # Repeat the latest 12-month seasonal pattern and apply recent trend.
        last_year = series.iloc[-12:].copy()
        if len(last_year) < 12:
            last_year = pd.Series([series.iloc[-1]] * 12,
                                  index=pd.date_range(series.index[-1],
                                                      periods=12, freq="MS"))
        slope = (series.iloc[-1] - series.iloc[-min(12, len(series))]) / max(1, min(12, len(series))-1)
        future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(),
                                     periods=FORECAST_PERIODS, freq="MS")
        forecast = pd.Series(
            [max(0, last_year.iloc[i % 12] + slope * (i + 1)) for i in range(FORECAST_PERIODS)],
            index=future_index
        )
        return forecast, "Seasonal-naive + trend fallback"


def calculate_holdout_accuracy(series):
    """Simple 6-month holdout check for forecasting accuracy."""
    if len(series) < 18:
        return None
    train, test = series.iloc[:-6], series.iloc[-6:]
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fitted = ARIMA(train, order=(1, 1, 1)).fit()
        pred = fitted.forecast(6)
    except Exception:
        pred = pd.Series([train.iloc[-1]] * 6, index=test.index)

    mae = (test - pred).abs().mean()
    rmse = (((test - pred) ** 2).mean()) ** 0.5
    mape = ((test - pred).abs() / test.abs()).mean() * 100
    return float(mae), float(rmse), float(mape)


def save_plot(history, forecast, method):
    plt.figure(figsize=(11, 6))
    plt.plot(history.index, history.values, label="Historical Revenue")
    plt.plot(forecast.index, forecast.values, linestyle="--", label=f"{method} Forecast")
    plt.axvline(history.index[-1], linestyle=":", label="Forecast Start")
    plt.title("Business Revenue Forecast - Day 37")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=160)
    plt.close()


def save_report(history, forecast, method, accuracy):
    growth = ((forecast.iloc[-1] / history.iloc[-1]) - 1) * 100
    avg_forecast = forecast.mean()

    lines = [
        "# Business Forecasting Report - Day 37",
        "",
        "## Objective",
        "Forecast the next 12 months of business revenue from historical monthly revenue data.",
        "",
        "## Method",
        f"Primary model: **{method}**.",
        "ARIMA is used when `statsmodels` is available. The script includes a deterministic fallback so the workflow remains runnable in minimal environments.",
        "",
        "## Forecast Summary",
        f"- Forecast horizon: **12 months**",
        f"- Average forecast revenue: **{avg_forecast:,.2f}**",
        f"- Last historical revenue: **{history.iloc[-1]:,.2f}**",
        f"- Final forecast revenue: **{forecast.iloc[-1]:,.2f}**",
        f"- Change from last historical month to final forecast: **{growth:.2f}%**",
        "",
        "## Historical Holdout Accuracy",
    ]
    if accuracy:
        mae, rmse, mape = accuracy
        lines += [
            f"- MAE: **{mae:,.2f}**",
            f"- RMSE: **{rmse:,.2f}**",
            f"- MAPE: **{mape:.2f}%**",
            "",
            "Lower MAE/RMSE/MAPE indicates better predictive accuracy on the holdout period."
        ]
    else:
        lines.append("- Not enough observations for a 6-month holdout evaluation.")
    lines += [
        "",
        "## Business Implications",
        "- Revenue forecasts support budgeting, staffing, inventory/capacity planning, and investment decisions.",
        "- Forecast uncertainty should be considered before committing large resources.",
        "- Forecasts should be refreshed as new monthly revenue becomes available.",
        "- Model accuracy should be monitored using holdout or rolling-origin evaluation rather than relying only on visual fit.",
        "",
        "## Limitations",
        "- This demonstration uses historical revenue only and does not include promotions, pricing, macroeconomic variables, or customer-level drivers.",
        "- A single ARIMA specification may not be optimal for every business; model selection and backtesting are recommended for production use.",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_data()
    history = df["revenue"].dropna()

    forecast, method = fit_forecast(history)
    forecast_df = forecast.rename("forecast_revenue").to_frame()
    forecast_df.to_csv(FORECAST_FILE, index_label="date")

    save_plot(history, forecast, method)
    accuracy = calculate_holdout_accuracy(history)
    save_report(history, forecast, method, accuracy)

    print(f"Method: {method}")
    print(f"Forecast saved to: {FORECAST_FILE}")
    print(f"Visualization saved to: {PLOT_FILE}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
