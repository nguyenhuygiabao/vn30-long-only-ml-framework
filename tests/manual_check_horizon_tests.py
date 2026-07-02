from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.horizon_tests import (
    HORIZON_PREDICTIONS_PATH,
    HORIZON_RESULTS_PATH,
    HORIZON_LABELS,
)


TOLERANCE: float = 1e-8


REQUIRED_RESULT_COLUMNS: list[str] = [
    "forecast_horizon_days",
    "horizon_label",
    "evaluated_dates",
    "feature_count",
    "average_rank_ic",
    "average_top_5_hit_rate",
    "average_top_5_actual_return",
    "average_selected_count",
    "average_after_cost_return",
    "return_volatility",
    "diagnostic_sharpe",
    "max_active_drawdown",
    "average_turnover",
    "maximum_turnover",
    "final_cumulative_after_cost_active_return",
]


def main() -> None:
    predictions_path = Path(HORIZON_PREDICTIONS_PATH)
    results_path = Path(HORIZON_RESULTS_PATH)

    predictions_file_exists = predictions_path.exists()
    results_file_exists = results_path.exists()

    predictions = pd.read_parquet(predictions_path)
    results = pd.read_csv(results_path)

    expected_horizons = sorted(HORIZON_LABELS.keys())
    prediction_horizons = sorted(
        predictions["forecast_horizon_days"].unique().tolist()
    )
    result_horizons = sorted(
        results["forecast_horizon_days"].unique().tolist()
    )

    required_result_columns_present = all(
        column in results.columns
        for column in REQUIRED_RESULT_COLUMNS
    )

    expected_prediction_horizons_present = (
        prediction_horizons == expected_horizons
    )

    expected_result_horizons_present = (
        result_horizons == expected_horizons
    )

    one_result_row_per_horizon = (
        results["forecast_horizon_days"].is_unique
        and len(results) == len(expected_horizons)
    )

    prediction_rows_valid = len(predictions) > 0

    evaluated_dates_valid = results["evaluated_dates"].min() > 0

    feature_counts_valid = results["feature_count"].min() > 0

    hit_rates_valid = (
        results["average_top_5_hit_rate"].between(0, 1).all()
    )

    sharpe_values_valid = results["diagnostic_sharpe"].notna().all()

    drawdowns_valid = results["max_active_drawdown"].le(0).all()

    turnover_valid = (
        results["average_turnover"].ge(0).all()
        and results["maximum_turnover"].le(0.40 + TOLERANCE).all()
    )

    best_ic_horizon = int(
        results.sort_values(
            "average_rank_ic",
            ascending=False,
        )["forecast_horizon_days"].iloc[0]
    )

    best_sharpe_horizon = int(
        results.sort_values(
            "diagnostic_sharpe",
            ascending=False,
        )["forecast_horizon_days"].iloc[0]
    )

    best_final_return_horizon = int(
        results.sort_values(
            "final_cumulative_after_cost_active_return",
            ascending=False,
        )["forecast_horizon_days"].iloc[0]
    )

    ten_day_is_best_on_core_metrics = (
        best_ic_horizon == 10
        and best_sharpe_horizon == 10
        and best_final_return_horizon == 10
    )

    print("Horizon prediction rows:", len(predictions))
    print("Horizon result rows:", len(results))
    print("Prediction horizons:", prediction_horizons)
    print("Result horizons:", result_horizons)
    print("\nHorizon results:")
    print(results.round(6).to_string(index=False))

    print("\nPredictions file exists:", predictions_file_exists)
    print("Results file exists:", results_file_exists)
    print("Required result columns present:", required_result_columns_present)
    print(
        "Expected prediction horizons present:",
        expected_prediction_horizons_present,
    )
    print(
        "Expected result horizons present:",
        expected_result_horizons_present,
    )
    print("One result row per horizon:", one_result_row_per_horizon)
    print("Prediction rows valid:", prediction_rows_valid)
    print("Evaluated dates valid:", evaluated_dates_valid)
    print("Feature counts valid:", feature_counts_valid)
    print("Hit rates valid:", hit_rates_valid)
    print("Sharpe values valid:", sharpe_values_valid)
    print("Drawdowns valid:", drawdowns_valid)
    print("Turnover valid:", turnover_valid)
    print("Best IC horizon:", best_ic_horizon)
    print("Best Sharpe horizon:", best_sharpe_horizon)
    print("Best final return horizon:", best_final_return_horizon)
    print("10d is best on core metrics:", ten_day_is_best_on_core_metrics)


if __name__ == "__main__":
    main()