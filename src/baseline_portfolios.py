from __future__ import annotations

import pandas as pd

from src.baselines import load_baseline_dataset
from src.walk_forward_split import TARGET_COLUMN


BASELINE_RETURNS_PATH: str = "data/processed/baseline_returns.parquet"

MOMENTUM_SCORE_COLUMN: str = "score_portfolio_momentum"
REVERSAL_SCORE_COLUMN: str = "score_portfolio_reversal"
LOW_VOLATILITY_SCORE_COLUMN: str = "score_portfolio_low_volatility"

STOCK_RETURN_COLUMN: str = "forward_return_5d"
BENCHMARK_RETURN_COLUMN: str = "vn30_forward_return_5d"

POSITION_COLUMNS: list[str] = [
    "date",
    "ticker",
    "strategy",
    "weight",
    STOCK_RETURN_COLUMN,
    BENCHMARK_RETURN_COLUMN,
    TARGET_COLUMN,
]


def add_portfolio_scores(data: pd.DataFrame) -> pd.DataFrame:
    scored_data = data.copy()

    if TARGET_COLUMN not in scored_data.columns:
        raise ValueError(
            f"Missing required target column: {TARGET_COLUMN}"
        )

    if "return_10d" in scored_data.columns:
        scored_data[MOMENTUM_SCORE_COLUMN] = scored_data["return_10d"]
    else:
        scored_data[MOMENTUM_SCORE_COLUMN] = pd.NA

    if "distance_from_20d_high" in scored_data.columns:
        scored_data[REVERSAL_SCORE_COLUMN] = -scored_data["distance_from_20d_high"]
    else:
        scored_data[REVERSAL_SCORE_COLUMN] = pd.NA

    if "rolling_vol_20d" in scored_data.columns:
        scored_data[LOW_VOLATILITY_SCORE_COLUMN] = -scored_data["rolling_vol_20d"]
    else:
        scored_data[LOW_VOLATILITY_SCORE_COLUMN] = pd.NA

    return scored_data

def select_equal_weight_all(
    data: pd.DataFrame,
    strategy_name: str = "equal_weight_all",
) -> pd.DataFrame:
    eligible_data = data.dropna(
        subset=[
            TARGET_COLUMN,
        ]
    ).copy()

    if eligible_data.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)

    eligible_data["strategy"] = strategy_name

    eligible_data["weight"] = (
        1.0
        / eligible_data.groupby("date")["ticker"].transform("count")
    )

    return eligible_data[POSITION_COLUMNS].copy()


def select_top_n_by_score(
    data: pd.DataFrame,
    score_column: str,
    top_n: int,
    strategy_name: str,
) -> pd.DataFrame:
    eligible_data = data.dropna(
        subset=[
            score_column,
            TARGET_COLUMN,
        ]
    ).copy()

    if eligible_data.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)

    selected_positions = (
        eligible_data.sort_values(
            by=[
                "date",
                score_column,
                "ticker",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .groupby("date", group_keys=False)
        .head(top_n)
        .copy()
    )

    selected_positions["strategy"] = strategy_name

    selected_positions["weight"] = (
        1.0
        / selected_positions.groupby("date")["ticker"].transform("count")
    )

    return selected_positions[POSITION_COLUMNS].copy()


def calculate_portfolio_returns(
    selected_positions: pd.DataFrame,
) -> pd.DataFrame:
    if selected_positions.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "strategy",
                "portfolio_forward_return_5d",
                "vn30_forward_return_5d",
                "active_return_vs_vn30_5d",
                "selected_count",
            ]
        )

    returns = (
        selected_positions.assign(
            weighted_stock_return=(
                selected_positions["weight"]
                * selected_positions[STOCK_RETURN_COLUMN]
            ),
            weighted_benchmark_return=(
                selected_positions["weight"]
                * selected_positions[BENCHMARK_RETURN_COLUMN]
            ),
            weighted_active_return=(
                selected_positions["weight"]
                * selected_positions[TARGET_COLUMN]
            ),
        )
        .groupby(
            [
                "date",
                "strategy",
            ],
            as_index=False,
        )
        .agg(
            portfolio_forward_return_5d=("weighted_stock_return", "sum"),
            vn30_forward_return_5d=("weighted_benchmark_return", "sum"),
            active_return_vs_vn30_5d=("weighted_active_return", "sum"),
            selected_count=("ticker", "count"),
        )
    )

    return returns


def build_baseline_positions(
    data: pd.DataFrame,
) -> pd.DataFrame:
    scored_data = add_portfolio_scores(data)

    equal_weight_positions = select_equal_weight_all(
        data=scored_data,
        strategy_name="equal_weight_all",
    )

    momentum_positions = select_top_n_by_score(
        data=scored_data,
        score_column=MOMENTUM_SCORE_COLUMN,
        top_n=5,
        strategy_name="top5_momentum",
    )

    reversal_positions = select_top_n_by_score(
        data=scored_data,
        score_column=REVERSAL_SCORE_COLUMN,
        top_n=5,
        strategy_name="top5_reversal",
    )

    low_volatility_positions = select_top_n_by_score(
        data=scored_data,
        score_column=LOW_VOLATILITY_SCORE_COLUMN,
        top_n=10,
        strategy_name="low_volatility_top10",
    )

    position_tables = [
        equal_weight_positions,
        momentum_positions,
        reversal_positions,
        low_volatility_positions,
    ]

    non_empty_position_tables = [
        position_table
        for position_table in position_tables
        if not position_table.empty
    ]

    if not non_empty_position_tables:
        return pd.DataFrame(columns=POSITION_COLUMNS)

    all_positions = pd.concat(
        non_empty_position_tables,
        ignore_index=True,
    )

    return all_positions

def build_baseline_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    baseline_positions = build_baseline_positions(data)

    baseline_returns = calculate_portfolio_returns(
        selected_positions=baseline_positions,
    )

    return baseline_returns


def compare_against_vn30_index(
    baseline_returns: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_returns.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "strategy",
                "portfolio_forward_return_5d",
                "vn30_forward_return_5d",
                "active_return_vs_vn30_5d",
                "selected_count",
            ]
        )

    comparison_columns = [
        "date",
        "strategy",
        "portfolio_forward_return_5d",
        "vn30_forward_return_5d",
        "active_return_vs_vn30_5d",
        "selected_count",
    ]

    return baseline_returns[comparison_columns].copy()


def main() -> None:
    modeling_dataset, historical_rows, prediction_rows = load_baseline_dataset()

    baseline_returns = build_baseline_returns(historical_rows)

    baseline_comparison = compare_against_vn30_index(
        baseline_returns=baseline_returns,
    )

    baseline_returns.to_parquet(
        BASELINE_RETURNS_PATH,
        index=False,
    )

    print("Modeling rows:", len(modeling_dataset))
    print("Historical rows:", len(historical_rows))
    print("Prediction-only rows:", len(prediction_rows))
    print("Baseline return rows:", len(baseline_returns))
    print("Baseline returns path:", BASELINE_RETURNS_PATH)

    print("\nStrategies evaluated:")
    print(sorted(baseline_returns["strategy"].unique().tolist()))

    print("\nBaseline return summary:")
    print(
        baseline_returns.groupby("strategy", as_index=False).agg(
            evaluated_dates=("date", "count"),
            average_portfolio_forward_return_5d=("portfolio_forward_return_5d", "mean"),
            average_active_return_vs_vn30_5d=("active_return_vs_vn30_5d", "mean"),
            average_selected_count=("selected_count", "mean"),
        )
    )

    print("\nComparison against VN30 index:")
    print(baseline_comparison)


if __name__ == "__main__":
    main()
