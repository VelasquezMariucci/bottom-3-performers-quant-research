from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------
# Study settings
# --------------------------------------------------

START_DATE = "2018-06-19"
FORECAST_DATE = "2026-07-08"

# yfinance treats the end date as exclusive.
# July 9 is used so that July 8 is included.
END_DATE = (pd.Timestamp(FORECAST_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

OUTPUT_FOLDER = Path("data/sectors")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Sector ETFs and benchmark
# --------------------------------------------------

SECTORS = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLK": "Information Technology",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}

SYMBOLS = {
    **SECTORS,
    "SPY": "S&P 500 Benchmark",
}


# --------------------------------------------------
# Download raw market data
# --------------------------------------------------

all_data = []
tickers = list(SYMBOLS.keys())

print("Downloading all sector ETFs and SPY...")

downloaded_data = yf.download(
    tickers=tickers,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=False,
    actions=True,
    group_by="ticker",
    threads=False,
    progress=True,
    timeout=30,
)

if downloaded_data.empty:
    raise RuntimeError(
        "Yahoo Finance returned no data. "
        "You may still be rate-limited. Wait and try again later."
    )

missing_tickers = []

for ticker, sector_name in SYMBOLS.items():
    if ticker not in downloaded_data.columns.get_level_values(0):
        missing_tickers.append(ticker)
        continue

    history = downloaded_data[ticker].copy()

    if "Close" not in history.columns:
        missing_tickers.append(ticker)
        continue

    history = history.dropna(subset=["Close"])

    if history.empty:
        missing_tickers.append(ticker)
        continue

    history.index = pd.to_datetime(history.index)

    if history.index.tz is not None:
        history.index = history.index.tz_convert(None)

    history.index = history.index.normalize()
    history.index.name = "Date"

    history = history.reset_index()

    history.insert(1, "Ticker", ticker)
    history.insert(2, "Sector", sector_name)

    history = history.rename(
        columns={
            "Adj Close": "Adjusted_Close",
            "Stock Splits": "Stock_Splits",
            "Capital Gains": "Capital_Gains",
        }
    )

    required_columns = [
        "Date",
        "Ticker",
        "Sector",
        "Open",
        "High",
        "Low",
        "Close",
        "Adjusted_Close",
        "Volume",
        "Dividends",
        "Stock_Splits",
        "Capital_Gains",
    ]

    for column in required_columns:
        if column not in history.columns:
            history[column] = 0.0

    all_data.append(history[required_columns])

if missing_tickers:
    raise RuntimeError(
        "Yahoo Finance did not return data for: " + ", ".join(missing_tickers)
    )

raw_data = pd.concat(all_data, ignore_index=True)
raw_data = raw_data.sort_values(["Date", "Ticker"])

raw_data.to_csv(
    OUTPUT_FOLDER / "sector_daily_raw.csv",
    index=False,
)


# --------------------------------------------------
# Adjusted prices
# --------------------------------------------------

adjusted_prices = raw_data.pivot(
    index="Date",
    columns="Ticker",
    values="Adjusted_Close",
).sort_index()

# Keep only dates available for every ETF.
adjusted_prices = adjusted_prices.dropna(how="any")

adjusted_prices.to_csv(OUTPUT_FOLDER / "adjusted_close_prices.csv")


# --------------------------------------------------
# Daily returns
# --------------------------------------------------

daily_returns = adjusted_prices.pct_change(fill_method=None)
daily_returns = daily_returns.dropna(how="all")

daily_returns.to_csv(OUTPUT_FOLDER / "daily_returns.csv")


# --------------------------------------------------
# Monthly prices and returns
# --------------------------------------------------

monthly_prices = adjusted_prices.groupby(adjusted_prices.index.to_period("M")).last()

monthly_prices.index = monthly_prices.index.to_timestamp("M")

monthly_returns = monthly_prices.pct_change(fill_method=None)

monthly_prices.to_csv(OUTPUT_FOLDER / "monthly_adjusted_prices.csv")

monthly_returns.to_csv(OUTPUT_FOLDER / "monthly_returns.csv")


# --------------------------------------------------
# Returns relative to the S&P 500
# --------------------------------------------------

sector_daily_returns = daily_returns[list(SECTORS)]
spy_daily_returns = daily_returns["SPY"]

daily_relative_returns = sector_daily_returns.subtract(
    spy_daily_returns,
    axis=0,
)

daily_relative_returns.to_csv(OUTPUT_FOLDER / "daily_returns_vs_spy.csv")

sector_monthly_returns = monthly_returns[list(SECTORS)]
spy_monthly_returns = monthly_returns["SPY"]

monthly_relative_returns = sector_monthly_returns.subtract(
    spy_monthly_returns,
    axis=0,
)

monthly_relative_returns.to_csv(OUTPUT_FOLDER / "monthly_returns_vs_spy.csv")


# --------------------------------------------------
# RSI calculation
# --------------------------------------------------


def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    price_change = prices.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / periods,
        adjust=False,
        min_periods=periods,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / periods,
        adjust=False,
        min_periods=periods,
    ).mean()

    relative_strength = average_gain / average_loss

    return 100 - (100 / (1 + relative_strength))


# --------------------------------------------------
# Build sector indicators
# --------------------------------------------------

feature_frames = []

spy_prices = adjusted_prices["SPY"]
spy_returns = spy_prices.pct_change(fill_method=None)

for ticker, sector_name in SECTORS.items():
    prices = adjusted_prices[ticker]
    returns = prices.pct_change(fill_method=None)

    ticker_raw = (
        raw_data.loc[raw_data["Ticker"] == ticker]
        .set_index("Date")
        .reindex(adjusted_prices.index)
    )

    features = pd.DataFrame(index=adjusted_prices.index)

    features["Ticker"] = ticker
    features["Sector"] = sector_name
    features["Adjusted_Close"] = prices
    features["Volume"] = ticker_raw["Volume"]

    # Absolute returns
    features["Return_1D"] = returns
    features["Return_1M"] = prices.pct_change(21)
    features["Return_3M"] = prices.pct_change(63)
    features["Return_6M"] = prices.pct_change(126)
    features["Return_12M"] = prices.pct_change(252)

    # Momentum: previous 12 months excluding the latest month
    features["Momentum_12_1"] = prices.shift(21) / prices.shift(252) - 1

    # Annualised volatility
    features["Volatility_1M"] = returns.rolling(21).std() * np.sqrt(252)

    features["Volatility_3M"] = returns.rolling(63).std() * np.sqrt(252)

    features["Volatility_6M"] = returns.rolling(126).std() * np.sqrt(252)

    # Technical indicators
    features["RSI_14D"] = calculate_rsi(prices)

    features["Price_vs_MA50"] = prices / prices.rolling(50).mean() - 1

    features["Price_vs_MA200"] = prices / prices.rolling(200).mean() - 1

    # Loss from the previous historical high
    features["Drawdown"] = prices / prices.cummax() - 1

    # Relative performance against SPY
    for days, label in [
        (21, "1M"),
        (63, "3M"),
        (126, "6M"),
        (252, "12M"),
    ]:
        sector_return = prices.pct_change(days)
        spy_return = spy_prices.pct_change(days)

        features[f"Relative_Return_{label}"] = sector_return - spy_return

    # Rolling beta and correlation with SPY
    features["Beta_1Y"] = (
        returns.rolling(252).cov(spy_returns) / spy_returns.rolling(252).var()
    )

    features["Correlation_SPY_1Y"] = returns.rolling(252).corr(spy_returns)

    # Trading activity
    features["Average_Volume_1M"] = features["Volume"].rolling(21).mean()

    features["Volume_vs_1M_Average"] = (
        features["Volume"] / features["Average_Volume_1M"]
    )

    feature_frames.append(features.reset_index())


sector_features = pd.concat(
    feature_frames,
    ignore_index=True,
)

sector_features.to_csv(
    OUTPUT_FOLDER / "sector_features.csv",
    index=False,
)


# --------------------------------------------------
# Data-quality summary
# --------------------------------------------------

quality_summary = (
    raw_data.groupby(["Ticker", "Sector"])
    .agg(
        First_Date=("Date", "min"),
        Last_Date=("Date", "max"),
        Observations=("Date", "count"),
        Missing_Adjusted_Prices=("Adjusted_Close", lambda x: x.isna().sum()),
    )
    .reset_index()
)

quality_summary.to_csv(
    OUTPUT_FOLDER / "data_quality_summary.csv",
    index=False,
)


# --------------------------------------------------
# Final confirmation
# --------------------------------------------------

print("\nDownload complete.")
print(f"Common start date: {adjusted_prices.index.min().date()}")
print(f"Final date: {adjusted_prices.index.max().date()}")
print(f"Common trading days: {len(adjusted_prices):,}")
print(f"Files saved in: {OUTPUT_FOLDER.resolve()}")

print("\nFiles created:")

for file_path in sorted(OUTPUT_FOLDER.glob("*.csv")):
    print(f"  - {file_path.name}")
