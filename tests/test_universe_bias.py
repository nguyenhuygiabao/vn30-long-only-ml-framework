from __future__ import annotations

import pandas as pd
import pytest

from src.universe_bias import (
    assert_complete_membership_interval_data,
    build_membership_interval_audit,
    detect_current_only_history_bias,
)


def membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
            "effective_from": [
                "2020-01-01",
                "2020-01-05",
                "2020-01-01",
                "2020-01-01",
            ],
            "effective_to": [
                "2020-01-02",
                "",
                "",
                "2020-01-02",
            ],
        }
    )


def test_interval_audit_preserves_exit_and_reentry() -> None:
    market = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-05",
                "2020-01-01",
                "2020-01-01",
            ],
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
        }
    )

    audit = build_membership_interval_audit(
        market,
        membership(),
        end_date="2020-01-06",
    )

    aaa = audit.loc[audit["ticker"].eq("AAA")]

    assert len(aaa) == 2
    assert not audit["missing_interval"].any()
    assert_complete_membership_interval_data(audit)


def test_interval_audit_fails_when_reentry_data_are_absent() -> None:
    market = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
            ],
            "ticker": ["AAA", "BBB", "CCC"],
        }
    )

    audit = build_membership_interval_audit(
        market,
        membership(),
        end_date="2020-01-06",
    )

    with pytest.raises(ValueError, match="no market data"):
        assert_complete_membership_interval_data(audit)


def test_current_only_vendor_history_is_detected() -> None:
    market = pd.DataFrame(
        {
            "date": ["2020-01-06", "2020-01-06"],
            "ticker": ["AAA", "BBB"],
        }
    )

    assert detect_current_only_history_bias(
        market,
        membership(),
        as_of_date="2020-01-06",
    )


def test_duplicate_market_keys_are_rejected() -> None:
    market = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "ticker": ["AAA", "AAA"],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_membership_interval_audit(
            market,
            membership(),
        )