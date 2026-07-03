from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import pandas as pd

SCENARIO_ORDER = [
    ("herding_aware", "normal"),
    ("herding_aware", "price_limit_aware"),
    ("normal", "normal"),
    ("normal", "price_limit_aware"),
]

SCENARIO_LABELS = {
    ("herding_aware", "normal"): "Herding-aware / Normal",
    ("herding_aware", "price_limit_aware"): "Herding-aware / Price-limit aware",
    ("normal", "normal"): "Normal / Normal",
    ("normal", "price_limit_aware"): "Normal / Price-limit aware",
}

SCENARIO_COLORS = {
    ("herding_aware", "normal"): "#1f77b4",
    ("herding_aware", "price_limit_aware"): "#ff7f0e",
    ("normal", "normal"): "#2ca02c",
    ("normal", "price_limit_aware"): "#d62728",
}


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

def display_scenario_label(
    optimization_mode: str,
    execution_mode: str,
) -> str:
    return SCENARIO_LABELS.get(
        (optimization_mode, execution_mode),
        f"{optimization_mode} / {execution_mode}",
    )


def place_legend_outside() -> None:
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )


def scenario_color(
    optimization_mode: str,
    execution_mode: str,
) -> str:
    return SCENARIO_COLORS.get(
        (optimization_mode, execution_mode),
        "#333333",
    )

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

    plt.figure(figsize=(15, 6))

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

        label = display_scenario_label(optimization_mode, execution_mode)
        color = scenario_color(optimization_mode, execution_mode)

        plt.plot(
            group["date"],
            group["cumulative_after_cost_active_return"],
            label=label,
            color=color,
            linewidth=0.9,
            alpha=0.85,
        )

    plt.title("Cumulative After-Cost Active Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative after-cost active return")
    place_legend_outside()
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

    scenario_order = [
        ("herding_aware", "normal"),
        ("herding_aware", "price_limit_aware"),
        ("normal", "normal"),
        ("normal", "price_limit_aware"),
    ]

    scenario_labels = {
        ("herding_aware", "normal"): "Herding-aware / Normal execution",
        ("herding_aware", "price_limit_aware"): "Herding-aware / Price-limit aware",
        ("normal", "normal"): "Normal / Normal execution",
        ("normal", "price_limit_aware"): "Normal / Price-limit aware",
    }

    scenario_colors = {
        ("herding_aware", "normal"): "#0072B2",
        ("herding_aware", "price_limit_aware"): "#E69F00",
        ("normal", "normal"): "#009E73",
        ("normal", "price_limit_aware"): "#CC79A7",
    }

    scenario_series = []
    global_min = 0.0

    for optimization_mode, execution_mode in scenario_order:
        group = data[
            data["optimization_mode"].eq(optimization_mode)
            & data["execution_mode"].eq(execution_mode)
        ].copy()

        if group.empty:
            continue

        group = group.sort_values("date").copy()
        cumulative_return = group["cumulative_after_cost_active_return"]
        running_peak = cumulative_return.cummax()
        group["active_drawdown"] = cumulative_return - running_peak

        weekly_worst_drawdown = (
            group.set_index("date")["active_drawdown"]
            .resample("W-FRI")
            .min()
            .dropna()
        )

        if weekly_worst_drawdown.empty:
            continue

        global_min = min(global_min, weekly_worst_drawdown.min())

        scenario_series.append(
            {
                "key": (optimization_mode, execution_mode),
                "label": scenario_labels[(optimization_mode, execution_mode)],
                "color": scenario_colors[(optimization_mode, execution_mode)],
                "series": weekly_worst_drawdown,
                "worst_value": weekly_worst_drawdown.min(),
                "worst_date": weekly_worst_drawdown.idxmin(),
            }
        )

    if not scenario_series:
        raise ValueError("No drawdown series available for plotting.")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    y_min = global_min * 1.10 if global_min < 0 else -0.05
    y_max = 0.02

    for ax, item in zip(axes, scenario_series):
        series = item["series"]

        ax.plot(
            series.index,
            series.values,
            color=item["color"],
            linewidth=1.0,
        )
        ax.fill_between(
            series.index,
            series.values,
            0,
            color=item["color"],
            alpha=0.12,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(item["label"], fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)

        ax.scatter(
            [item["worst_date"]],
            [item["worst_value"]],
            color=item["color"],
            s=24,
            zorder=3,
        )

        ax.annotate(
            f"{item['worst_value']:.3f}",
            xy=(item["worst_date"], item["worst_value"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    for ax in axes[2:]:
        ax.set_xlabel("Date")

    axes[0].set_ylabel("Weekly worst active drawdown")
    axes[2].set_ylabel("Weekly worst active drawdown")

    fig.suptitle("Weekly Worst Active Drawdown by Scenario", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

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

    plt.figure(figsize=(16, 6))

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

        weekly_average_turnover = (
            group.set_index("date")["portfolio_turnover"]
            .resample("W-FRI")
            .mean()
            .dropna()
        )

        label = display_scenario_label(optimization_mode, execution_mode)
        color = scenario_color(optimization_mode, execution_mode)

        plt.plot(
            weekly_average_turnover.index,
            weekly_average_turnover,
            label=label,
            linewidth=0.7,
            alpha=0.75,
        )

    plt.title("Weekly Average Portfolio Turnover")
    plt.xlabel("Date")
    plt.ylabel("Average weekly turnover")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    place_legend_outside()
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

    plt.figure(figsize=(15, 6))

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

        label = display_scenario_label(optimization_mode, execution_mode)
        color = scenario_color(optimization_mode, execution_mode)

        plt.plot(
            group["date"],
            group["rolling_diagnostic_sharpe"],
            label=label,
            color=color,
            linewidth=0.9,
            alpha=0.85,
        )

    plt.title(f"Rolling {rolling_window}-Window Diagnostic Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Rolling diagnostic Sharpe")
    place_legend_outside()
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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