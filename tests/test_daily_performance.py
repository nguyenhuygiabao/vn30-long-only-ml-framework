from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from src.paper_trading.daily_performance import (
    build_daily_performance_row,
    equal_weight_benchmark_return,
)


def broker(
    *,
    cash: str = "500",
    quantity: int = 5,
):
    return SimpleNamespace(
        settled_cash=Decimal(cash),
        unsettled_cash=Decimal("0"),
        positions={
            "FPT": SimpleNamespace(
                economic_quantity=quantity,
            )
        },
    )


def test_equal_weight_benchmark_return() -> None:
    result = equal_weight_benchmark_return(
        previous_closes={
            "FPT": "100",
            "HPG": "200",
        },
        current_closes={
            "FPT": "110",
            "HPG": "180",
        },
    )

    assert result == Decimal("0")


def test_initial_performance_row_starts_at_zero_return() -> None:
    row = build_daily_performance_row(
        performance_date="2026-07-22",
        broker=broker(),
        mark_prices={"FPT": "100"},
        previous_rows=pd.DataFrame(),
        benchmark_return="0.01",
        turnover="0.20",
        skipped_trade_count=3,
    )

    assert row["portfolio_value"] == "1000"
    assert row["market_value"] == "500"
    assert row["daily_return"] == "0"
    assert row["cumulative_return"] == "0"
    assert row["cash_weight"] == "0.5"
    assert row["holdings_count"] == 1


def test_subsequent_performance_compounds_returns() -> None:
    previous = pd.DataFrame([
        {
            "portfolio_value": "1000",
            "cumulative_return": "0",
            "benchmark_value": "1000",
            "cumulative_benchmark_return": "0",
        }
    ])

    row = build_daily_performance_row(
        performance_date="2026-07-23",
        broker=broker(cash="500", quantity=6),
        mark_prices={"FPT": "100"},
        previous_rows=previous,
        benchmark_return="0.02",
        turnover="0",
        skipped_trade_count=0,
    )

    assert row["portfolio_value"] == "1100"
    assert row["daily_return"] == "0.1"
    assert row["cumulative_return"] == "0.1"
    assert row["benchmark_value"] == "1020.00"
    assert row["active_return"] == "0.08"
    assert row["drawdown"] == "0"


def test_drawdown_uses_running_portfolio_peak() -> None:
    previous = pd.DataFrame([
        {
            "portfolio_value": "1000",
            "cumulative_return": "0",
            "benchmark_value": "1000",
            "cumulative_benchmark_return": "0",
        },
        {
            "portfolio_value": "1200",
            "cumulative_return": "0.2",
            "benchmark_value": "1010",
            "cumulative_benchmark_return": "0.01",
        },
    ])

    row = build_daily_performance_row(
        performance_date="2026-07-24",
        broker=broker(cash="500", quantity=5),
        mark_prices={"FPT": "100"},
        previous_rows=previous,
        benchmark_return="0",
        turnover="0",
        skipped_trade_count=0,
    )

    assert row["portfolio_value"] == "1000"
    assert row["drawdown"] == str(
        Decimal("1000") / Decimal("1200") - Decimal("1")
    )


def test_missing_benchmark_prices_fails() -> None:
    with pytest.raises(
        ValueError,
        match="No valid benchmark returns",
    ):
        equal_weight_benchmark_return({}, {})
