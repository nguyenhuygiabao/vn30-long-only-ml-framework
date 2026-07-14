from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.model_candidates import (
    DEFAULT_REGIME_POLICY,
    build_historical_market_regimes,
    build_rank_ensemble_history,
)


def _turnover(previous: pd.Series, target: pd.Series) -> tuple[float, float, float]:
    tickers = previous.index.union(target.index)
    old = previous.reindex(tickers, fill_value=0.0)
    new = target.reindex(tickers, fill_value=0.0)
    buys = (new - old).clip(lower=0.0).sum()
    sells = (old - new).clip(upper=0.0).abs().sum()
    return float(0.5 * (new - old).abs().sum()), float(buys), float(sells)


def _apply_turnover_cap(
    previous: pd.Series,
    target: pd.Series,
    max_turnover: float,
) -> pd.Series:
    turnover, _, _ = _turnover(previous, target)
    if turnover <= max_turnover or turnover == 0.0:
        return target[target > 1e-12]

    tickers = previous.index.union(target.index)
    old = previous.reindex(tickers, fill_value=0.0)
    desired = target.reindex(tickers, fill_value=0.0)
    adjusted = old + (max_turnover / turnover) * (desired - old)
    return adjusted[adjusted > 1e-12]


def build_non_overlapping_policy_returns(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    policy: Mapping[str, str] | None = None,
    top_n: int = 8,
    holding_period_days: int = 10,
    settlement_lag_days: int = 2,
    target_exposure: float = 0.97,
    max_turnover: float = 0.25,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    sell_tax_rate: float = 0.001,
) -> pd.DataFrame:
    """Backtest a fixed regime policy on non-overlapping forward-return labels.

    Rebalances are spaced by ``holding_period_days`` observed trading dates.  If
    that period is at least the settlement lag, proceeds from a prior rebalance
    would settle before the next rebalance; this is reported as a T+2-compatible
    schedule, not as a reconstruction of historical broker fills.
    """
    if top_n <= 0 or holding_period_days <= 0 or settlement_lag_days < 0:
        raise ValueError("top_n and holding_period_days must be positive")
    if not 0.0 <= target_exposure <= 1.0 or not 0.0 < max_turnover <= 1.0:
        raise ValueError("Exposure and turnover inputs are outside valid bounds")

    policy_map = dict(DEFAULT_REGIME_POLICY if policy is None else policy)
    expected_regimes = {"trend_up", "trend_down", "high_volatility"}
    missing = sorted(expected_regimes.difference(policy_map))
    if missing:
        raise ValueError(f"Regime policy is missing selections: {missing}")

    ensemble = build_rank_ensemble_history(predictions)
    combined = pd.concat([predictions.copy(), ensemble], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    combined["ticker"] = combined["ticker"].astype(str).str.strip().str.upper()
    available_models = set(combined["model_name"])
    invalid = sorted(
        {
            model for model in policy_map.values()
            if model != "cash" and model not in available_models
        }
    )
    if invalid:
        raise ValueError(f"Regime policy selects unavailable models: {invalid}")

    regimes = build_historical_market_regimes(market_data)
    complete_actuals = (
        combined.groupby(["date", "ticker"])["actual_return"]
        .agg(lambda values: values.notna().all())
        .groupby(level="date")
        .all()
    )
    model_dates = complete_actuals.loc[complete_actuals].rename("complete_actuals")
    schedule = regimes.merge(
        model_dates,
        left_on="date",
        right_index=True,
        how="inner",
    ).sort_values("date")
    if schedule.empty:
        raise ValueError("No regime dates overlap available predictions")
    schedule = schedule.iloc[::holding_period_days].copy()
    schedule["selected_model"] = schedule["market_regime"].map(policy_map)

    previous = pd.Series(dtype=float)
    rows: list[dict[str, object]] = []

    for schedule_row in schedule.itertuples(index=False):
        date = schedule_row.date
        selected_model = schedule_row.selected_model
        daily_actual_returns = (
            combined.loc[combined["date"] == date]
            .groupby("ticker")["actual_return"]
            .first()
            .dropna()
        )
        unavailable_previous = previous.index.difference(daily_actual_returns.index)
        forced_exit_weight = float(previous.reindex(unavailable_previous).sum())
        carried_previous = previous.drop(unavailable_previous)

        if selected_model == "cash":
            target = pd.Series(dtype=float)
        else:
            daily = combined.loc[
                (combined["date"] == date)
                & (combined["model_name"] == selected_model)
            ].nlargest(top_n, "predicted_return")
            if len(daily) < top_n:
                raise ValueError(f"Insufficient predictions for {selected_model} on {date}")
            target = pd.Series(
                target_exposure / top_n,
                index=daily["ticker"],
                dtype=float,
            )

        if not target.index.isin(daily_actual_returns.index).all():
            raise ValueError(f"Selected targets lack realized returns on {date}")

        forced_exit_turnover = 0.5 * forced_exit_weight
        optional_turnover_capacity = max(max_turnover - forced_exit_turnover, 0.0)
        if optional_turnover_capacity > 0.0:
            weights = _apply_turnover_cap(
                carried_previous,
                target,
                max_turnover=optional_turnover_capacity,
            )
        else:
            weights = carried_previous
        turnover, buys, sells = _turnover(previous, weights)
        realized = daily_actual_returns.reindex(weights.index)
        if realized.isna().any():
            raise ValueError(f"Missing realized returns for selected holdings on {date}")
        before_cost = float((weights * realized.to_numpy()).sum())
        total_cost = (
            (buys + sells) * (commission_rate + slippage_rate)
            + sells * sell_tax_rate
        )
        rows.append(
            {
                "date": date,
                "market_regime": schedule_row.market_regime,
                "selected_model": selected_model,
                "before_cost_return": before_cost,
                "total_cost": total_cost,
                "after_cost_return": before_cost - total_cost,
                "portfolio_turnover": turnover,
                "total_weight": float(weights.sum()),
                "selected_count": len(weights),
                "forced_exit_weight": forced_exit_weight,
                "turnover_cap_breached_by_forced_exit": (
                    turnover > max_turnover + 1e-12
                ),
                "settlement_compatible": holding_period_days >= settlement_lag_days,
            }
        )
        previous = weights

    history = pd.DataFrame(rows)
    history["cumulative_after_cost_return"] = (1.0 + history["after_cost_return"]).cumprod() - 1.0
    return history


def summarize_non_overlapping_policy_returns(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        raise ValueError("Policy backtest history is empty")
    returns = history["after_cost_return"]
    volatility = returns.std()
    return pd.DataFrame(
        [
            {
                "rebalance_dates": history["date"].nunique(),
                "average_after_cost_return": returns.mean(),
                "return_volatility": volatility,
                "diagnostic_sharpe": returns.mean() / volatility if volatility else np.nan,
                "average_turnover": history["portfolio_turnover"].mean(),
                "maximum_turnover": history["portfolio_turnover"].max(),
                "forced_exit_dates": int((history["forced_exit_weight"] > 0.0).sum()),
                "maximum_forced_exit_weight": history["forced_exit_weight"].max(),
                "final_cumulative_after_cost_return": history[
                    "cumulative_after_cost_return"
                ].iloc[-1],
                "settlement_compatible": bool(history["settlement_compatible"].all()),
            }
        ]
    )
