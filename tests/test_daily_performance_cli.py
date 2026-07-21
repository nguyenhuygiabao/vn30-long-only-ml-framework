from __future__ import annotations

from decimal import Decimal

import pandas as pd

from scripts.record_daily_paper_performance import (
    _execution_turnover,
    _skipped_trade_count,
    parse_args,
)


def test_performance_cli_defaults_to_preview() -> None:
    args = parse_args([
        "--date",
        "2026-07-21",
    ])

    assert args.date == "2026-07-21"
    assert args.write is False


def test_performance_cli_requires_explicit_write() -> None:
    args = parse_args([
        "--date",
        "2026-07-21",
        "--write",
    ])

    assert args.write is True


def test_execution_turnover_uses_requested_date() -> None:
    executions = pd.DataFrame([
        {
            "execution_date": "2026-07-21",
            "gross_value": "200",
        },
        {
            "execution_date": "2026-07-22",
            "gross_value": "300",
        },
    ])

    result = _execution_turnover(
        executions=executions,
        performance_date=pd.Timestamp(
            "2026-07-21"
        ).date(),
        reference_value=Decimal("1000"),
    )

    assert result == Decimal("0.2")


def test_skipped_trade_count_uses_requested_date() -> None:
    skipped = pd.DataFrame([
        {"date": "2026-07-21"},
        {"date": "2026-07-21"},
        {"date": "2026-07-22"},
    ])

    result = _skipped_trade_count(
        skipped=skipped,
        performance_date=pd.Timestamp(
            "2026-07-21"
        ).date(),
    )

    assert result == 2
