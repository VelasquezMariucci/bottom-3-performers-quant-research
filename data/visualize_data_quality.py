from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_FILE = Path("data/sectors/sector_daily_raw.csv")
OUTPUT_FOLDER = Path("data/sectors/raw_visualize")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

EXPECTED_TICKERS = [
    "XLC",
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLV",
    "XLI",
    "XLK",
    "XLB",
    "XLRE",
    "XLU",
    "SPY",
]


# --------------------------------------------------
# Load and prepare the data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adjusted_Close",
    "Volume",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# --------------------------------------------------
# Quality checks
# --------------------------------------------------

df["Duplicate_Row"] = df.duplicated(
    subset=["Date", "Ticker"],
    keep=False,
)

df["Missing_Price"] = (
    df[["Open", "High", "Low", "Close", "Adjusted_Close"]].isna().any(axis=1)
)

df["Nonpositive_Price"] = (
    df[["Open", "High", "Low", "Close", "Adjusted_Close"]] <= 0
).any(axis=1)

df["Invalid_Volume"] = df["Volume"].isna() | (df["Volume"] < 0)

df["OHLC_Issue"] = (
    (df["High"] < df["Low"])
    | (df["High"] < df[["Open", "Close"]].max(axis=1))
    | (df["Low"] > df[["Open", "Close"]].min(axis=1))
)

df["Return_1D"] = df.groupby("Ticker")["Adjusted_Close"].pct_change(fill_method=None)

# A 15% daily move is a review threshold, not a deletion rule.
df["Large_Return"] = df["Return_1D"].abs() > 0.15

# Unchanged prices can reveal stale observations.
df["Unchanged_Price"] = df.groupby("Ticker")["Adjusted_Close"].diff().eq(0)


# --------------------------------------------------
# Missing trading dates
# --------------------------------------------------

all_dates = pd.DatetimeIndex(sorted(df["Date"].dropna().unique()))

coverage = (
    df.assign(Available=1)
    .pivot_table(
        index="Ticker",
        columns="Date",
        values="Available",
        aggfunc="max",
    )
    .reindex(index=EXPECTED_TICKERS, columns=all_dates)
    .fillna(0)
)

missing_dates = coverage.eq(0).sum(axis=1)

observations_per_date = (
    df.groupby("Date")["Ticker"].nunique().reindex(all_dates, fill_value=0)
)


# --------------------------------------------------
# Summary and rows requiring review
# --------------------------------------------------

quality_summary = (
    df.groupby("Ticker")
    .agg(
        First_Date=("Date", "min"),
        Last_Date=("Date", "max"),
        Rows=("Date", "size"),
        Duplicate_Rows=("Duplicate_Row", "sum"),
        Missing_Prices=("Missing_Price", "sum"),
        Nonpositive_Prices=("Nonpositive_Price", "sum"),
        Invalid_Volumes=("Invalid_Volume", "sum"),
        OHLC_Issues=("OHLC_Issue", "sum"),
        Large_Returns=("Large_Return", "sum"),
        Unchanged_Prices=("Unchanged_Price", "sum"),
    )
    .reindex(EXPECTED_TICKERS)
)

quality_summary["Missing_Dates"] = missing_dates

quality_summary.to_csv(OUTPUT_FOLDER / "data_quality_summary.csv")

issue_columns = [
    "Duplicate_Row",
    "Missing_Price",
    "Nonpositive_Price",
    "Invalid_Volume",
    "OHLC_Issue",
    "Large_Return",
]

rows_to_review = df[df[issue_columns].any(axis=1)].copy()

rows_to_review.to_csv(
    OUTPUT_FOLDER / "rows_to_review.csv",
    index=False,
)


# --------------------------------------------------
# Visualization
# --------------------------------------------------

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10),
    constrained_layout=True,
)

# 1. Data availability heatmap
ax = axes[0, 0]

ax.imshow(
    coverage.values,
    aspect="auto",
    interpolation="nearest",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
)

ax.set_yticks(range(len(coverage.index)))
ax.set_yticklabels(coverage.index)

tick_positions = (
    pd.Series(range(len(all_dates))).iloc[:: max(1, len(all_dates) // 6)].to_numpy()
)

ax.set_xticks(tick_positions)
ax.set_xticklabels(
    all_dates[tick_positions].strftime("%Y-%m"),
    rotation=45,
    ha="right",
)

ax.set_title("Data availability: green = present, red = missing")
ax.set_xlabel("Date")
ax.set_ylabel("Ticker")


# 2. Number of symbols available each day
ax = axes[0, 1]

ax.plot(
    observations_per_date.index,
    observations_per_date.values,
)

ax.axhline(
    len(EXPECTED_TICKERS),
    color="green",
    linestyle="--",
    label=f"Expected: {len(EXPECTED_TICKERS)}",
)

ax.set_title("Number of ETFs available per trading date")
ax.set_xlabel("Date")
ax.set_ylabel("ETFs available")
ax.legend()


# 3. Normalized adjusted prices
ax = axes[1, 0]

prices = df.pivot_table(
    index="Date",
    columns="Ticker",
    values="Adjusted_Close",
    aggfunc="last",
).reindex(columns=EXPECTED_TICKERS)

normalized_prices = prices.apply(lambda series: series / series.dropna().iloc[0] * 100)

normalized_prices.plot(
    ax=ax,
    linewidth=1,
)

ax.set_title("Adjusted prices normalized to 100")
ax.set_xlabel("Date")
ax.set_ylabel("Normalized value")
ax.legend(
    ncol=3,
    fontsize=8,
)


# 4. Daily-return distributions
ax = axes[1, 1]

return_data = df[["Ticker", "Return_1D"]].dropna()

sns.boxplot(
    data=return_data,
    x="Ticker",
    y="Return_1D",
    order=EXPECTED_TICKERS,
    showfliers=True,
    ax=ax,
)

ax.axhline(0.15, color="red", linestyle="--", linewidth=1)
ax.axhline(-0.15, color="red", linestyle="--", linewidth=1)

ax.set_title("Daily-return distributions")
ax.set_xlabel("Ticker")
ax.set_ylabel("Daily return")
ax.tick_params(axis="x", rotation=45)

dashboard_path = OUTPUT_FOLDER / "data_quality_dashboard.png"

fig.savefig(
    dashboard_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nData-quality summary:")
print(quality_summary.to_string())

print(f"\nRows requiring review: {len(rows_to_review):,}")
print(f"Dashboard: {dashboard_path.resolve()}")
print(
    "Detailed issues:",
    (OUTPUT_FOLDER / "rows_to_review.csv").resolve(),
)
