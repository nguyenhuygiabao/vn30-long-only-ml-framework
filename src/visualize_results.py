from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd


BACKTEST_RETURNS_PATH: str = "data/processed/backtest_returns.parquet"
TREE_FEATURE_IMPORTANCE_PATH: str = "data/processed/tree_feature_importance.parquet"
ABLATION_RESULTS_PATH: str = "reports/tables/ablation_results.csv"
HORIZON_RESULTS_PATH: str = "reports/tables/horizon_results.csv"
FIGURES_DIR: str = "reports/figures"


def print_dataset_overview(
    name: str,
    path: str,
) -> None:
    file_path = Path(path)

    print("\n" + "=" * 80)
    print(name)
    print("Path:", file_path)
    print("Exists:", file_path.exists())

    if not file_path.exists():
        return

    if file_path.suffix == ".parquet":
        data = pd.read_parquet(file_path)
    else:
        data = pd.read_csv(file_path)

    print("Rows:", len(data))
    print("Columns:")
    for column in data.columns:
        print("-", column)

def plot_cumulative_after_cost_returns() -> Path:
    data = pd.read_parquet(BACKTEST_RETURNS_PATH)
    data["date"] = pd.to_datetime(data["date"])

    required_columns = [
        "optimization_mode",
        "execution_mode",
        "date",
        "cumulative_after_cost_active_return",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for cumulative return plot: {missing_columns}"
        )

    output_path = Path(FIGURES_DIR) / "cumulative_after_cost_active_return.png"

    plt.figure(figsize=(12, 6))

    for (
        optimization_mode,
        execution_mode,
    ), group in data.groupby(
        [
            "optimization_mode",
            "execution_mode",
        ]
    ):
        group = group.sort_values("date")

        label = f"{optimization_mode} | {execution_mode}"

        plt.plot(
            group["date"],
            group["cumulative_after_cost_active_return"],
            label=label,
        )

    plt.title("Cumulative After-Cost Active Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative after-cost active return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def plot_active_drawdowns() -> Path:
    data = pd.read_parquet(BACKTEST_RETURNS_PATH)
    data["date"] = pd.to_datetime(data["date"])

    required_columns = [
        "optimization_mode",
        "execution_mode",
        "date",
        "cumulative_after_cost_active_return",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for drawdown plot: {missing_columns}"
        )

    output_path = Path(FIGURES_DIR) / "active_drawdown.png"

    plt.figure(figsize=(12, 6))

    for (
        optimization_mode,
        execution_mode,
    ), group in data.groupby(
        [
            "optimization_mode",
            "execution_mode",
        ]
    ):
        group = group.sort_values("date").copy()

        running_peak = group[
            "cumulative_after_cost_active_return"
        ].cummax()

        group["active_drawdown"] = (
            group["cumulative_after_cost_active_return"]
            - running_peak
        )

        label = f"{optimization_mode} | {execution_mode}"

        plt.plot(
            group["date"],
            group["active_drawdown"],
            label=label,
        )

    plt.title("Active Drawdown from Previous Peak")
    plt.xlabel("Date")
    plt.ylabel("Active drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def plot_portfolio_turnover() -> Path:
    data = pd.read_parquet(BACKTEST_RETURNS_PATH)
    data["date"] = pd.to_datetime(data["date"])

    required_columns = [
        "optimization_mode",
        "execution_mode",
        "date",
        "portfolio_turnover",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for turnover plot: {missing_columns}"
        )

    output_path = Path(FIGURES_DIR) / "portfolio_turnover.png"

    plt.figure(figsize=(12, 6))

    for (
        optimization_mode,
        execution_mode,
    ), group in data.groupby(
        [
            "optimization_mode",
            "execution_mode",
        ]
    ):
        group = group.sort_values("date")

        label = f"{optimization_mode} | {execution_mode}"

        plt.plot(
            group["date"],
            group["portfolio_turnover"],
            label=label,
        )

    plt.title("Portfolio Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def plot_top_tree_feature_importance(top_n: int = 20) -> Path:
    data = pd.read_parquet(TREE_FEATURE_IMPORTANCE_PATH)

    required_columns = [
        "model_name",
        "feature",
        "importance",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for feature importance plot: {missing_columns}"
        )

    model_data = data[data["model_name"] == "gradient_boosting"].copy()

    if model_data.empty:
        raise ValueError("No gradient_boosting feature importance rows found")

    feature_importance = (
        model_data.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    output_path = Path(FIGURES_DIR) / "top_gradient_boosting_feature_importance.png"

    plt.figure(figsize=(10, 8))

    plt.barh(
        feature_importance["feature"],
        feature_importance["importance"],
    )

    plt.title("Top Gradient Boosting Feature Importance")
    plt.xlabel("Average feature importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def plot_ablation_diagnostic_sharpe() -> Path:
    data = pd.read_csv(ABLATION_RESULTS_PATH)

    required_columns = [
        "ablation_name",
        "diagnostic_sharpe",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for ablation plot: {missing_columns}"
        )

    plot_data = data.sort_values(
        "diagnostic_sharpe",
        ascending=True,
    )

    output_path = Path(FIGURES_DIR) / "ablation_diagnostic_sharpe.png"

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data["ablation_name"],
        plot_data["diagnostic_sharpe"],
    )

    plt.title("Feature Ablation: Diagnostic Sharpe")
    plt.xlabel("Diagnostic Sharpe")
    plt.ylabel("Ablation setting")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def plot_horizon_diagnostic_sharpe() -> Path:
    data = pd.read_csv(HORIZON_RESULTS_PATH)

    required_columns = [
        "forecast_horizon_days",
        "diagnostic_sharpe",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for horizon plot: {missing_columns}"
        )

    plot_data = data.sort_values("forecast_horizon_days").copy()
    plot_data["horizon_label_short"] = (
        plot_data["forecast_horizon_days"].astype(str) + "d"
    )

    output_path = Path(FIGURES_DIR) / "horizon_diagnostic_sharpe.png"

    plt.figure(figsize=(8, 6))

    plt.bar(
        plot_data["horizon_label_short"],
        plot_data["diagnostic_sharpe"],
    )

    plt.title("Forecast Horizon Comparison: Diagnostic Sharpe")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Diagnostic Sharpe")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def plot_rolling_diagnostic_sharpe(
    rolling_window: int = 60,
) -> Path:
    data = pd.read_parquet(BACKTEST_RETURNS_PATH)
    data["date"] = pd.to_datetime(data["date"])

    required_columns = [
        "optimization_mode",
        "execution_mode",
        "date",
        "after_cost_return",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for rolling Sharpe plot: {missing_columns}"
        )

    output_path = Path(FIGURES_DIR) / "rolling_diagnostic_sharpe.png"

    plt.figure(figsize=(12, 6))

    for (
        optimization_mode,
        execution_mode,
    ), group in data.groupby(
        [
            "optimization_mode",
            "execution_mode",
        ]
    ):
        group = group.sort_values("date").copy()

        rolling_mean = group["after_cost_return"].rolling(
            rolling_window
        ).mean()

        rolling_volatility = group["after_cost_return"].rolling(
            rolling_window
        ).std()

        group["rolling_diagnostic_sharpe"] = (
            rolling_mean / rolling_volatility
        )

        label = f"{optimization_mode} | {execution_mode}"

        plt.plot(
            group["date"],
            group["rolling_diagnostic_sharpe"],
            label=label,
        )

    plt.title(f"Rolling {rolling_window}-Window Diagnostic Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Rolling diagnostic Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def plot_horizon_rank_ic() -> Path:
    data = pd.read_csv(HORIZON_RESULTS_PATH)

    required_columns = [
        "forecast_horizon_days",
        "average_rank_ic",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for horizon IC plot: {missing_columns}"
        )

    plot_data = data.sort_values("forecast_horizon_days").copy()
    plot_data["horizon_label_short"] = (
        plot_data["forecast_horizon_days"].astype(str) + "d"
    )

    output_path = Path(FIGURES_DIR) / "horizon_rank_ic.png"

    plt.figure(figsize=(8, 6))

    plt.bar(
        plot_data["horizon_label_short"],
        plot_data["average_rank_ic"],
    )

    plt.title("Forecast Horizon Comparison: Average Rank IC")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Average rank IC")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def main() -> None:
    Path(FIGURES_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    print_dataset_overview(
        name="Backtest returns",
        path=BACKTEST_RETURNS_PATH,
    )

    print_dataset_overview(
        name="Tree feature importance",
        path=TREE_FEATURE_IMPORTANCE_PATH,
    )

    print_dataset_overview(
        name="Ablation results",
        path=ABLATION_RESULTS_PATH,
    )

    print_dataset_overview(
        name="Horizon results",
        path=HORIZON_RESULTS_PATH,
    )
    cumulative_return_path = plot_cumulative_after_cost_returns()
    print("\nSaved figure:", cumulative_return_path)
    drawdown_path = plot_active_drawdowns()
    print("Saved figure:", drawdown_path)
    turnover_path = plot_portfolio_turnover()
    print("Saved figure:", turnover_path)
    feature_importance_path = plot_top_tree_feature_importance()
    print("Saved figure:", feature_importance_path)
    ablation_path = plot_ablation_diagnostic_sharpe()
    print("Saved figure:", ablation_path)
    horizon_path = plot_horizon_diagnostic_sharpe()
    print("Saved figure:", horizon_path)
    rolling_sharpe_path = plot_rolling_diagnostic_sharpe()
    print("Saved figure:", rolling_sharpe_path)
    horizon_rank_ic_path = plot_horizon_rank_ic()
    print("Saved figure:", horizon_rank_ic_path)

if __name__ == "__main__":
    main()