import pandas as pd

from src.baseline_portfolios import (
    BENCHMARK_RETURN_COLUMN,
    STOCK_RETURN_COLUMN,
    build_baseline_positions,
    build_baseline_returns,
    compare_against_vn30_index,
)
from src.walk_forward_split import TARGET_COLUMN


synthetic_rows = []

for date in [
    pd.Timestamp("2024-01-02"),
    pd.Timestamp("2024-01-09"),
]:
    for ticker_number in range(1, 13):
        ticker = f"T{ticker_number:02d}"

        vn30_forward_return = 0.02
        relative_forward_return = ticker_number / 1000
        stock_forward_return = vn30_forward_return + relative_forward_return

        synthetic_rows.append(
            {
                "date": date,
                "ticker": ticker,
                "return_10d": (13 - ticker_number) / 100,
                "distance_from_20d_high": -(13 - ticker_number) / 100,
                "rolling_vol_20d": ticker_number / 100,
                STOCK_RETURN_COLUMN: stock_forward_return,
                BENCHMARK_RETURN_COLUMN: vn30_forward_return,
                TARGET_COLUMN: relative_forward_return,
            }
        )

synthetic_data = pd.DataFrame(synthetic_rows)

baseline_positions = build_baseline_positions(synthetic_data)

baseline_returns = build_baseline_returns(synthetic_data)

baseline_comparison = compare_against_vn30_index(
    baseline_returns=baseline_returns,
)

print("Synthetic input rows:", len(synthetic_data))
print("Baseline position rows:", len(baseline_positions))
print("Baseline return rows:", len(baseline_returns))

print("\nPosition count by strategy:")
print(
    baseline_positions.groupby("strategy", as_index=False).agg(
        selected_positions=("ticker", "count"),
        evaluated_dates=("date", "nunique"),
        average_weight=("weight", "mean"),
    )
)

print("\nReturn count by strategy:")
print(
    baseline_returns.groupby("strategy", as_index=False).agg(
        evaluated_dates=("date", "count"),
        average_selected_count=("selected_count", "mean"),
    )
)

print("\nBaseline returns:")
print(baseline_returns)

print("\nComparison against VN30 index:")
print(baseline_comparison)

expected_position_counts = {
    "equal_weight_all": 24,
    "top5_momentum": 10,
    "top5_reversal": 10,
    "low_volatility_top10": 20,
}

actual_position_counts = (
    baseline_positions.groupby("strategy")["ticker"]
    .count()
    .to_dict()
)

position_counts_correct = actual_position_counts == expected_position_counts

return_rows_correct = len(baseline_returns) == 8

comparison_rows_correct = len(baseline_comparison) == 8

print("\nPosition counts correct:", position_counts_correct)
print("Return rows correct:", return_rows_correct)
print("Comparison rows correct:", comparison_rows_correct)
