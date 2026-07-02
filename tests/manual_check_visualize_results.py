from __future__ import annotations

from pathlib import Path

from src.visualize_results import FIGURES_DIR


EXPECTED_FIGURES: list[str] = [
    "cumulative_after_cost_active_return.png",
    "active_drawdown.png",
    "portfolio_turnover.png",
    "top_gradient_boosting_feature_importance.png",
    "ablation_diagnostic_sharpe.png",
    "horizon_diagnostic_sharpe.png",
    "rolling_diagnostic_sharpe.png",
    "horizon_rank_ic.png",
]


def main() -> None:
    figures_dir = Path(FIGURES_DIR)

    figure_checks = []

    for figure_name in EXPECTED_FIGURES:
        figure_path = figures_dir / figure_name

        figure_checks.append(
            {
                "figure": figure_name,
                "exists": figure_path.exists(),
                "size_bytes": figure_path.stat().st_size
                if figure_path.exists()
                else 0,
            }
        )

    all_figures_exist = all(
        check["exists"]
        for check in figure_checks
    )

    all_figures_non_empty = all(
        check["size_bytes"] > 0
        for check in figure_checks
    )

    print("Figures directory:", figures_dir)
    print("Expected figure count:", len(EXPECTED_FIGURES))

    print("\nFigure checks:")
    for check in figure_checks:
        print(
            check["figure"],
            "| exists:",
            check["exists"],
            "| size bytes:",
            check["size_bytes"],
        )

    print("\nAll figures exist:", all_figures_exist)
    print("All figures non-empty:", all_figures_non_empty)


if __name__ == "__main__":
    main()