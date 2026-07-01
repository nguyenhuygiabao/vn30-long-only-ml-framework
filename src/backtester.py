from __future__ import annotations

from pathlib import Path

import pandas as pd


OPTIMIZED_WEIGHTS_PATH: str = "data/processed/optimized_weights.parquet"
FEATURES_PATH: str = "data/processed/features_combined.parquet"
BACKTEST_RETURNS_PATH: str = "data/processed/backtest_returns.parquet"

COMMISSION_RATE: float = 0.001
SLIPPAGE_RATE: float = 0.001
LIQUIDITY_PENALTY_RATE: float = 0.001
LOW_LIQUIDITY_QUANTILE: float = 0.20

KEY_COLUMNS: list[str] = [
    "date",
    "ticker",
]


def load_optimized_weights(
    path: str = OPTIMIZED_WEIGHTS_PATH,
) -> pd.DataFrame:
    weights = pd.read_parquet(path)

    weights["date"] = pd.to_datetime(weights["date"])

    return weights


def load_liquidity_features(
    path: str = FEATURES_PATH,
) -> pd.DataFrame:
    features = pd.read_parquet(path)

    features["date"] = pd.to_datetime(features["date"])

    required_columns = [
        "date",
        "ticker",
        "average_daily_value_20d",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing liquidity feature columns: {missing_columns}")

    liquidity = features[required_columns].copy()

    daily_low_liquidity_cutoff = (
        liquidity.groupby("date")["average_daily_value_20d"]
        .quantile(LOW_LIQUIDITY_QUANTILE)
        .rename("low_liquidity_cutoff")
        .reset_index()
    )

    liquidity = liquidity.merge(
        daily_low_liquidity_cutoff,
        on="date",
        how="left",
    )

    liquidity["is_low_liquidity"] = (
        liquidity["average_daily_value_20d"]
        <= liquidity["low_liquidity_cutoff"]
    )

    return liquidity


def attach_liquidity_features(
    weights: pd.DataFrame,
    liquidity: pd.DataFrame,
) -> pd.DataFrame:
    return weights.merge(
        liquidity,
        on=KEY_COLUMNS,
        how="left",
    )


def calculate_backtest_returns(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    working = weights.copy()

    working["is_low_liquidity"] = working["is_low_liquidity"].eq(True)

    working["weighted_actual_return"] = (
        working["weight"] * working["actual_return"]
    )

    working["low_liquidity_weight_component"] = working["weight"].where(
        working["is_low_liquidity"],
        0.0,
    )

    daily_returns = working.groupby("date").agg(
        before_cost_return=("weighted_actual_return", "sum"),
        portfolio_turnover=("portfolio_turnover", "max"),
        selected_count=("ticker", "count"),
        total_weight=("weight", "sum"),
        low_liquidity_weight=("low_liquidity_weight_component", "sum"),
    )

    daily_returns["commission_cost"] = (
        daily_returns["portfolio_turnover"] * COMMISSION_RATE
    )

    daily_returns["slippage_cost"] = (
        daily_returns["portfolio_turnover"] * SLIPPAGE_RATE
    )

    daily_returns["liquidity_penalty"] = (
        daily_returns["portfolio_turnover"]
        * daily_returns["low_liquidity_weight"]
        * LIQUIDITY_PENALTY_RATE
    )

    daily_returns["total_cost"] = (
        daily_returns["commission_cost"]
        + daily_returns["slippage_cost"]
        + daily_returns["liquidity_penalty"]
    )

    daily_returns["after_cost_return"] = (
        daily_returns["before_cost_return"]
        - daily_returns["total_cost"]
    )

    daily_returns["cumulative_before_cost_active_return"] = (
        daily_returns["before_cost_return"].cumsum()
    )

    daily_returns["cumulative_after_cost_active_return"] = (
        daily_returns["after_cost_return"].cumsum()
    )

    return daily_returns.reset_index()


def save_backtest_returns(
    backtest_returns: pd.DataFrame,
    output_path: str = BACKTEST_RETURNS_PATH,
) -> str:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backtest_returns.to_parquet(
        path,
        index=False,
    )

    return str(path)


def main() -> None:
    weights = load_optimized_weights()
    liquidity = load_liquidity_features()
    weights_with_liquidity = attach_liquidity_features(weights, liquidity)
    backtest_returns = calculate_backtest_returns(weights_with_liquidity)
    output_path = save_backtest_returns(backtest_returns)

    print("Backtest returns path:", output_path)
    print("Backtest rows:", len(backtest_returns))
    print("Date count:", backtest_returns["date"].nunique())
    print("Average before-cost return:", backtest_returns["before_cost_return"].mean())
    print("Average after-cost return:", backtest_returns["after_cost_return"].mean())
    print("Average total cost:", backtest_returns["total_cost"].mean())
    print("Maximum turnover:", backtest_returns["portfolio_turnover"].max())
    print("Maximum total weight:", backtest_returns["total_weight"].max())
    print("Maximum low-liquidity weight:", backtest_returns["low_liquidity_weight"].max())
    print("Final cumulative before-cost active return:", backtest_returns["cumulative_before_cost_active_return"].iloc[-1])
    print("Final cumulative after-cost active return:", backtest_returns["cumulative_after_cost_active_return"].iloc[-1])

    print("\nSample:")
    print(backtest_returns.head(10))


if __name__ == "__main__":
    main()