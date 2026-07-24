from __future__ import annotations

import pandas as pd
import pytest

from src.universe_reconstruction import (
    reconstruct_constituent_history,
)


def initial_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
            ],
            "ticker": ["AAA", "BBB", "CCC"],
        }
    )


def review_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
            ]
        }
    )


def membership_changes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_date": [
                "2020-02-01",
                "2020-02-01",
            ],
            "action": ["remove", "add"],
            "ticker": ["AAA", "DDD"],
        }
    )


def source_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
            ],
            "publication_date": [
                "2019-12-25",
                "2020-01-25",
                "2020-02-25",
            ],
            "source_url": [
                "https://example.com/initial",
                "https://example.com/february",
                "https://example.com/march",
            ],
            "verified": [True, True, "verified"],
        }
    )


def reconstruct():
    return reconstruct_constituent_history(
        initial_snapshot=initial_snapshot(),
        review_calendar=review_calendar(),
        membership_changes=membership_changes(),
        source_manifest=source_manifest(),
        expected_size=3,
    )


def test_reconstructs_snapshots_and_zero_change_review() -> None:
    result = reconstruct()

    february = result.snapshots.loc[
        result.snapshots["effective_date"].eq(
            pd.Timestamp("2020-02-01")
        ),
        "ticker",
    ]
    march = result.snapshots.loc[
        result.snapshots["effective_date"].eq(
            pd.Timestamp("2020-03-01")
        ),
        "ticker",
    ]

    assert set(february) == {"BBB", "CCC", "DDD"}
    assert set(march) == {"BBB", "CCC", "DDD"}
    assert len(result.snapshots) == 9
    assert result.audit["constituent_count"].eq(3).all()
    assert result.audit["status"].eq("pass").all()
    assert result.audit["verified"].all()
    assert result.audit["additions"].tolist() == [0, 1, 0]
    assert result.audit["removals"].tolist() == [0, 1, 0]
    assert result.ticker_pool["ticker"].tolist() == [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
    ]


def test_membership_history_preserves_reentry() -> None:
    calendar = pd.DataFrame(
        {
            "effective_date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
            ]
        }
    )
    changes = pd.DataFrame(
        {
            "effective_date": [
                "2020-02-01",
                "2020-02-01",
                "2020-03-01",
                "2020-03-01",
            ],
            "action": [
                "remove",
                "add",
                "remove",
                "add",
            ],
            "ticker": [
                "AAA",
                "DDD",
                "DDD",
                "AAA",
            ],
        }
    )

    result = reconstruct_constituent_history(
        initial_snapshot(),
        calendar,
        changes,
        source_manifest(),
        expected_size=3,
    )

    aaa = result.membership_history.loc[
        result.membership_history["ticker"].eq("AAA")
    ]

    assert len(aaa) == 2
    assert set(aaa["effective_from"]) == {
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-03-01"),
    }


def test_rejects_wrong_initial_snapshot_size() -> None:
    invalid = initial_snapshot().iloc[:2].copy()

    with pytest.raises(
        ValueError,
        match="exactly 3 unique tickers",
    ):
        reconstruct_constituent_history(
            invalid,
            review_calendar(),
            membership_changes(),
            source_manifest(),
            expected_size=3,
        )


def test_rejects_unordered_review_calendar() -> None:
    invalid = pd.DataFrame(
        {
            "effective_date": [
                "2020-01-01",
                "2020-03-01",
                "2020-02-01",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        reconstruct_constituent_history(
            initial_snapshot(),
            invalid,
            membership_changes(),
            source_manifest(),
            expected_size=3,
        )


def test_rejects_missing_source_date() -> None:
    invalid = source_manifest().iloc[:2].copy()

    with pytest.raises(
        ValueError,
        match="missing review dates",
    ):
        reconstruct_constituent_history(
            initial_snapshot(),
            review_calendar(),
            membership_changes(),
            invalid,
            expected_size=3,
        )


def test_rejects_unverified_source() -> None:
    invalid = source_manifest()
    invalid.loc[
        invalid["effective_date"].eq("2020-02-01"),
        "verified",
    ] = False

    with pytest.raises(
        ValueError,
        match="unverified dates",
    ):
        reconstruct_constituent_history(
            initial_snapshot(),
            review_calendar(),
            membership_changes(),
            invalid,
            expected_size=3,
        )


def test_rejects_removal_of_non_member() -> None:
    invalid = membership_changes()
    invalid.loc[
        invalid["action"].eq("remove"),
        "ticker",
    ] = "ZZZ"

    with pytest.raises(
        ValueError,
        match="Cannot remove non-members",
    ):
        reconstruct_constituent_history(
            initial_snapshot(),
            review_calendar(),
            invalid,
            source_manifest(),
            expected_size=3,
        )


def test_rejects_addition_of_existing_member() -> None:
    invalid = membership_changes()
    invalid.loc[
        invalid["action"].eq("add"),
        "ticker",
    ] = "BBB"

    with pytest.raises(
        ValueError,
        match="Cannot add existing members",
    ):
        reconstruct_constituent_history(
            initial_snapshot(),
            review_calendar(),
            invalid,
            source_manifest(),
            expected_size=3,
        )


def test_rejects_change_that_breaks_constituent_count() -> None:
    invalid = membership_changes().loc[
        membership_changes()["action"].eq("remove")
    ].copy()

    with pytest.raises(
        ValueError,
        match="2 constituents instead of 3",
    ):
        reconstruct_constituent_history(
            initial_snapshot(),
            review_calendar(),
            invalid,
            source_manifest(),
            expected_size=3,
        )


def test_rejects_changes_outside_review_calendar() -> None:
    invalid = membership_changes()
    invalid.loc[0, "effective_date"] = "2020-04-01"

    with pytest.raises(
        ValueError,
        match="unknown review dates",
    ):
        reconstruct_constituent_history(
            initial_snapshot(),
            review_calendar(),
            invalid,
            source_manifest(),
            expected_size=3,
        )