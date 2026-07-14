from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.regime_policy_backtest import summarize_non_overlapping_policy_returns


def summarize_walk_forward_candidates(
    histories: Mapping[str, pd.DataFrame],
    holdout_start: str | pd.Timestamp,
) -> pd.DataFrame:
    """Summarize pre-defined candidates before and after a chronological split."""
    split_date = pd.Timestamp(holdout_start)
    rows: list[pd.DataFrame] = []

    for candidate_name, history in histories.items():
        dated = history.copy()
        dated["date"] = pd.to_datetime(dated["date"], errors="raise")
        for period, sample in (
            ("training", dated.loc[dated["date"] < split_date]),
            ("holdout", dated.loc[dated["date"] >= split_date]),
        ):
            if len(sample) < 2:
                raise ValueError(
                    f"{candidate_name} has fewer than two {period} observations"
                )
            summary = summarize_non_overlapping_policy_returns(sample)
            summary.insert(0, "period", period)
            summary.insert(0, "candidate_name", candidate_name)
            rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def select_training_candidate(
    walk_forward_summary: pd.DataFrame,
    metric: str = "diagnostic_sharpe",
) -> str:
    """Choose one candidate using only the training-period diagnostic."""
    required = {"candidate_name", "period", metric}
    missing = sorted(required.difference(walk_forward_summary.columns))
    if missing:
        raise ValueError(f"Walk-forward summary is missing columns: {missing}")

    training = walk_forward_summary.loc[
        walk_forward_summary["period"] == "training"
    ].dropna(subset=[metric])
    if training.empty:
        raise ValueError("No training candidate has a finite selection metric")

    ordered = training.sort_values([metric, "candidate_name"], ascending=[False, True])
    return str(ordered.iloc[0]["candidate_name"])
