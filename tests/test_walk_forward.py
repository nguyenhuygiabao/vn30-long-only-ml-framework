from __future__ import annotations

import pandas as pd

from src.walk_forward import (
    select_training_candidate,
    summarize_walk_forward_candidates,
)


def history(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2023-12-25", periods=len(values))
    cumulative = (1.0 + pd.Series(values)).cumprod() - 1.0
    return pd.DataFrame(
        {
            "date": dates,
            "after_cost_return": values,
            "portfolio_turnover": [0.10] * len(values),
            "target_exposure": [0.97] * len(values),
            "forced_exit_weight": [0.0] * len(values),
            "cumulative_after_cost_return": cumulative,
            "settlement_compatible": [True] * len(values),
        }
    )


def test_walk_forward_selection_uses_training_only() -> None:
    histories = {
        "baseline": history([0.01, 0.01, -0.01, -0.01, 0.04, 0.04]),
        "guarded": history([0.02, 0.02, -0.005, -0.005, -0.04, -0.04]),
    }
    summary = summarize_walk_forward_candidates(histories, "2023-12-29")

    assert select_training_candidate(summary) == "guarded"
    assert set(summary["period"]) == {"training", "holdout"}
    baseline_holdout = summary.loc[
        (summary["candidate_name"] == "baseline")
        & (summary["period"] == "holdout")
    ].iloc[0]
    assert round(baseline_holdout["final_cumulative_after_cost_return"], 10) == 0.0816
