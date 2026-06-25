from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_ohlcv_csv

MOMENTUM_HORIZONS: tuple[int, ...] = (1,3,5,10,20,60)
REVERSAL_WINDOW: int = 20
OUTPUT_PATH: str = "data/processed/features_momentum.parquet"

def add_momentum_features (df: pd.DataFrame) -> pd.DataFrame: 
    result = df.sort_values(["ticker", "date"]).copy()

    grouped_adjusted_close = result.groupby("ticker")["adjusted_close"]

    for horizon in MOMENTUM_HORIZONS:
        previous_adjusted_close = grouped_adjusted_close.shift(horizon)

        result[f"return_{horizon}d"] = (
            result["adjusted_close"]/previous_adjusted_close -1)

    rolling_high_20d = (
        result
        .groupby("ticker")["adjusted_close"]
        .transform(
            lambda series: series.rolling(
                window = REVERSAL_WINDOW,
                min_periods= REVERSAL_WINDOW,
            ).max()
        )
    )

    rolling_low_20d = (
        result
        .groupby("ticker")["adjusted_close"]
        .transform(
            lambda series: series.rolling(
                window = REVERSAL_WINDOW,
                min_periods= REVERSAL_WINDOW,
            ).min()
        )
    )

    result["distance_from_20d_high"] = (
        result["adjusted_close"]/rolling_high_20d -1
    )

    result["distance_from_20d_low"] = (
        result["adjusted_close"]/rolling_low_20d -1
    )

    result["negative_shock_1d"] = (
        result["return_1d"].clip(upper = 0)
    )

    result["positive_shock_1d"] = (
        result["return_1d"].clip(lower = 0)
    )


    return result

def plot_momentum_features(
        features: pd.DataFrame,
        ticker: str, 
        output_path: str
) -> None:
    ticker_data = (
        features.loc[features["ticker"] == ticker]
        .sort_values("date")
        .copy()
    )

    plot_columns = [
        "return_1d",
        "return_3d",
        "return_5d",
    ]

    ticker_data.plot(
        x = "date",
        y = plot_columns,
        figsize = (10, 6),
        marker = "o"
    )

    plt.title(f"{ticker} Momentum Features")
    plt.xlabel("Date")
    plt.ylabel("Historical return")
    plt.axhline(0, linewidth =1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    

def main() -> None:
    df = load_ohlcv_csv("sample_data/sample_ohlcv.csv")
    features = add_momentum_features(df)

    features.to_parquet(OUTPUT_PATH, index = False)

    momentum_columns = [
        f"return_{horizon}d"
        for horizon in MOMENTUM_HORIZONS
    ]

    valid_counts = (
        features
        .groupby("ticker")[momentum_columns]
        .count()
    )

    first_rows = (
        features
        .groupby("ticker", sort=False)
        .head(1)
        .loc[:, ["ticker", *momentum_columns]]
        .copy()
    )

    first_rows["all_first_returns_missing"] = (
        first_rows[momentum_columns]
        .isna()
        .all(axis=1)
    )

    print("Valid momentum values by ticker:")
    print(valid_counts)

    print("\nTicker boundary check:")
    print(
        first_rows[
            ["ticker", "all_first_returns_missing"]
        ].to_string(index=False)
    )

    reversal_columns = [ 
      "distance_from_20d_high",
     "distance_from_20d_low"
    ]

    reversal_valid_counts = (
        features
        .groupby("ticker")[reversal_columns]
        .count()
    )

    print("\nValid 20-day reversal values by ticker:")
    print(reversal_valid_counts)


    shock_columns = [
        "negative_shock_1d",
        "positive_shock_1d"
    ]

    shock_valid_counts = (
        features
        .groupby("ticker")[shock_columns]
        .count()
    )
    
    plot_path = "reports/figures/fpt_momentum_features.png"
    plot_momentum_features(
        features = features,
        ticker= "FPT",
        output_path = plot_path,
    )

    print("\nMomentum plot saved to:")
    print(plot_path)

    print("\nValid shock values by ticker")
    print(shock_valid_counts)

    negative_shock_valid = (
        features["negative_shock_1d"].dropna() <= 0
    ).all()

    positive_shock_valid = (
        features["positive_shock_1d"].dropna() >= 0
    ).all()

    shock_reconstruction_valid = (
        (
            features["negative_shock_1d"] + features["positive_shock_1d"] - features["return_1d"]
        ).dropna()
        .abs()
        .lt(1e-12)
        .all()
    )

    print("\nShock logic checks:")
    print("Negative shock is never positive:", negative_shock_valid)
    print("Positive shock is never negative:", positive_shock_valid)
    print("Shocks reconstruct return_1d:", shock_reconstruction_valid)    



if __name__ == "__main__":
    main()

