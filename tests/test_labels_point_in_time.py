from __future__ import annotations

import pandas as pd
import pytest

from src.labels import build_point_in_time_relative_labels


def test_peer_label_uses_only_signal_date_members() -> None:
    stock_data = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-01",
                "2020-01-02",
                "2020-01-01",
                "2020-01-02",
            ],
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "adjusted_close": [10.0, 11.0, 20.0, 20.0, 30.0, 60.0],
        }
    )
    membership = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "effective_from": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
            ],
            "effective_to": ["", "", ""],
        }
    )

    labels = build_point_in_time_relative_labels(
        stock_data,
        membership,
    )
    first_date = labels.loc[
        labels["date"].eq(pd.Timestamp("2020-01-01"))
    ]

    assert set(first_date["ticker"]) == {"AAA", "BBB"}

    aaa = first_date.loc[first_date["ticker"].eq("AAA")].iloc[0]
    assert aaa["vn30_forward_return_1d"] == pytest.approx(0.0)
    assert aaa["forward_relative_return_1d"] == pytest.approx(0.1)


def test_returns_are_built_before_membership_filtering() -> None:
    stock_data = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
            "ticker": ["AAA", "AAA", "AAA"],
            "adjusted_close": [5.0, 6.0, 9.0],
        }
    )
    membership = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "effective_from": ["2020-01-02"],
            "effective_to": [""],
        }
    )

    labels = build_point_in_time_relative_labels(
        stock_data,
        membership,
    )
    entry_row = labels.loc[
        labels["date"].eq(pd.Timestamp("2020-01-02"))
    ].iloc[0]

    assert entry_row["forward_return_1d"] == pytest.approx(0.5)