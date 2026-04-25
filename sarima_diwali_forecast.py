from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATASET_URL = (
    "https://raw.githubusercontent.com/sinjoysaha/sales-analysis/master/all_data.csv"
)
DEFAULT_ITEMS = [
    "USB-C Charging Cable",
    "Lightning Charging Cable",
    "AAA Batteries (4-pack)",
]
DIWALI_2019 = pd.Timestamp("2019-10-27")
DIWALI_WINDOW_DAYS = 7


def load_sales_data(source: str) -> pd.DataFrame:
    """Load and clean the public electronics-store transaction dataset."""
    data = pd.read_csv(source)

    data = data[data["Order Date"].notna()].copy()
    data = data[data["Order Date"].str.lower() != "order date"].copy()
    data["Order Date"] = pd.to_datetime(
        data["Order Date"], format="%m/%d/%y %H:%M", errors="coerce"
    )
    data["Quantity Ordered"] = pd.to_numeric(
        data["Quantity Ordered"], errors="coerce"
    )
    data = data.dropna(subset=["Order Date", "Product", "Quantity Ordered"])
    data["date"] = data["Order Date"].dt.normalize()

    return data


def daily_item_demand(data: pd.DataFrame, item: str) -> pd.Series:
    item_rows = data[data["Product"].eq(item)]
    if item_rows.empty:
        raise ValueError(f"No rows found for item: {item}")

    daily = item_rows.groupby("date")["Quantity Ordered"].sum().sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_index, fill_value=0).astype(float)


def fit_sarima(train: pd.Series):
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_item(data: pd.DataFrame, item: str, start: pd.Timestamp, end: pd.Timestamp):
    daily = daily_item_demand(data, item)
    train = daily[daily.index < start]
    if len(train) < 30:
        raise ValueError(f"Need at least 30 daily observations before {start.date()}.")

    horizon = (end - start).days + 1
    result = fit_sarima(train)
    forecast = result.get_forecast(steps=horizon)
    predicted = forecast.predicted_mean.clip(lower=0).round()
    conf_int = forecast.conf_int().clip(lower=0).round()
    predicted.index = pd.date_range(start, end, freq="D")
    conf_int.index = predicted.index

    actual = daily.reindex(predicted.index)
    output = pd.DataFrame(
        {
            "date": predicted.index.date,
            "item": item,
            "predicted_units": predicted.astype(int).to_numpy(),
            "lower_units": conf_int.iloc[:, 0].astype(int).to_numpy(),
            "upper_units": conf_int.iloc[:, 1].astype(int).to_numpy(),
            "actual_units": actual.fillna(0).astype(int).to_numpy(),
        }
    )
    output["model"] = "SARIMA(1,1,1)(1,0,1,7)"
    return output, result.aic


def build_forecast(source: str, items: list[str], output_path: Path) -> pd.DataFrame:
    data = load_sales_data(source)
    start = DIWALI_2019 - pd.Timedelta(days=DIWALI_WINDOW_DAYS)
    end = DIWALI_2019 + pd.Timedelta(days=DIWALI_WINDOW_DAYS)

    forecasts = []
    diagnostics = []
    for item in items:
        forecast, aic = forecast_item(data, item, start, end)
        forecasts.append(forecast)
        diagnostics.append({"item": item, "aic": round(float(aic), 2)})

    output = pd.concat(forecasts, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    diagnostics_path = output_path.with_name("sarima_diagnostics.csv")
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast three electronics item requirements for Diwali season."
    )
    parser.add_argument("--source", default=DATASET_URL, help="CSV path or URL.")
    parser.add_argument(
        "--items",
        nargs="+",
        default=DEFAULT_ITEMS,
        help="Product names to forecast.",
    )
    parser.add_argument(
        "--output",
        default="outputs/diwali_sarima_forecast.csv",
        help="Forecast CSV output path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    forecast_table = build_forecast(args.source, args.items, Path(args.output))
    summary = (
        forecast_table.groupby("item", as_index=False)["predicted_units"]
        .sum()
        .rename(columns={"predicted_units": "diwali_window_requirement"})
    )
    print(summary.to_string(index=False))
