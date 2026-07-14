from __future__ import annotations

import pandas as pd

from src.regime_policy_backtest import (
    build_non_overlapping_policy_returns,
    summarize_non_overlapping_policy_returns,
)


def market_data() -> pd.DataFrame:
    rows = []
    for ticker in ("AAA", "BBB", "CCC"):
        close = 100.0
        for date in pd.bdate_range("2025-01-01", periods=220):
            close *= 1.005 if date.day % 7 else 0.99
            rows.append({"date": date, "ticker": ticker, "adjusted_close": close})
    return pd.DataFrame(rows)


def predictions() -> pd.DataFrame:
    rows = []
    for date in pd.bdate_range("2025-07-01", periods=30):
        for model_name, scores in {
            "gradient_boosting": [0.3, 0.2, 0.1],
            "random_forest": [0.2, 0.3, 0.1],
        }.items():
            for ticker, score, actual in zip(
                ("AAA", "BBB", "CCC"), scores, (0.04, 0.02, -0.01)
            ):
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "model_name": model_name,
                        "predicted_return": score,
                        "actual_return": actual,
                    }
                )
    return pd.DataFrame(rows)


def test_non_overlapping_policy_backtest_caps_turnover_and_compounds() -> None:
    policy = {
        "trend_up": "random_forest",
        "trend_down": "cash",
        "high_volatility": "random_forest",
    }
    history = build_non_overlapping_policy_returns(
        predictions(),
        market_data(),
        policy=policy,
        top_n=2,
        holding_period_days=10,
        max_turnover=0.25,
    )
    summary = summarize_non_overlapping_policy_returns(history)

    assert len(history) >= 1
    assert history["portfolio_turnover"].max() <= 0.25
    assert history["settlement_compatible"].all()
    assert summary.loc[0, "rebalance_dates"] == len(history)


def test_non_overlapping_policy_skips_incomplete_realized_return_dates() -> None:
    incomplete = predictions()
    incomplete.loc[incomplete.index[0], "actual_return"] = float("nan")
    history = build_non_overlapping_policy_returns(
        incomplete,
        market_data(),
        top_n=2,
        holding_period_days=10,
    )

    assert not history.empty
    assert history["after_cost_return"].notna().all()


def test_non_overlapping_policy_forces_exit_when_held_ticker_disappears() -> None:
    complete = predictions()
    baseline = build_non_overlapping_policy_returns(
        complete,
        market_data(),
        policy={
            "trend_up": "gradient_boosting",
            "trend_down": "gradient_boosting",
            "high_volatility": "gradient_boosting",
        },
        top_n=2,
        holding_period_days=10,
    )
    assert len(baseline) >= 2

    second_date = baseline.iloc[1]["date"]
    missing_ticker = "AAA"
    reduced = complete.loc[
        ~((complete["date"] == second_date) & (complete["ticker"] == missing_ticker))
    ].copy()
    history = build_non_overlapping_policy_returns(
        reduced,
        market_data(),
        policy={
            "trend_up": "gradient_boosting",
            "trend_down": "gradient_boosting",
            "high_volatility": "gradient_boosting",
        },
        top_n=2,
        holding_period_days=10,
    )

    exit_row = history.loc[history["date"] == second_date].iloc[0]
    assert exit_row["forced_exit_weight"] > 0.0
    assert pd.notna(exit_row["after_cost_return"])
