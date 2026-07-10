from __future__ import annotations

from datetime import date

import pytest

from src.paper_trading.calendar import TradingCalendar


def test_t_plus_two_skips_weekend() -> None:
    calendar = TradingCalendar.from_weekdays("2026-07-01", "2026-07-20")

    assert calendar.settlement_date(date(2026, 7, 10), 2) == date(2026, 7, 14)


def test_t_plus_two_skips_explicit_holiday() -> None:
    calendar = TradingCalendar.from_weekdays(
        "2026-07-01",
        "2026-07-20",
        holidays=["2026-07-13"],
    )

    assert calendar.settlement_date("2026-07-10", 2) == date(2026, 7, 15)


def test_non_trading_trade_date_is_rejected() -> None:
    calendar = TradingCalendar.from_weekdays("2026-07-01", "2026-07-20")

    with pytest.raises(ValueError, match="not in the trading calendar"):
        calendar.settlement_date("2026-07-11", 2)


def test_calendar_must_cover_full_settlement_offset() -> None:
    calendar = TradingCalendar.from_weekdays("2026-07-01", "2026-07-10")

    with pytest.raises(ValueError, match="does not cover offset"):
        calendar.settlement_date("2026-07-10", 2)
