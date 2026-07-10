from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.paper_trading.schemas import LEDGER_SCHEMAS
from src.paper_trading.scoring import (
    DailyScoringResult,
    build_signal_ledger_rows,
    score_completed_market_data,
    score_latest_modeling_dataset,
)
from src.walk_forward_split import TARGET_COLUMN


TICKERS = ("FPT", "VCB", "VNM")


def modeling_dataset() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2026-06-01", periods=15)

    for date_index, market_date in enumerate(dates):
        for ticker_index, ticker in enumerate(TICKERS):
            rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "feature_one": date_index + ticker_index / 10,
                    "feature_two": (date_index + 1) * (ticker_index + 1),
                    TARGET_COLUMN: (ticker_index - 1) * 0.01 + date_index / 1000,
                }
            )

    frame = pd.DataFrame(rows)
    frame.loc[frame["date"] == dates[-1], TARGET_COLUMN] = pd.NA
    return frame


def completed_market_data() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2026-01-01", periods=80)

    for date_index, market_date in enumerate(dates):
        for ticker_index, ticker in enumerate(TICKERS):
            close = (
                50
                + ticker_index * 10
                + date_index * (0.05 + ticker_index * 0.01)
                + np.sin(date_index / 5 + ticker_index)
            )
            volume = 1_000_000 + date_index * 1000 + ticker_index * 100_000
            rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adjusted_close": close,
                    "volume": volume,
                    "value_traded": close * volume,
                }
            )

    return pd.DataFrame(rows)


def score(frame: pd.DataFrame | None = None) -> DailyScoringResult:
    return score_latest_modeling_dataset(
        modeling_dataset=modeling_dataset() if frame is None else frame,
        expected_tickers=TICKERS,
        horizon_days=2,
        minimum_training_dates=3,
    )


def test_latest_scoring_enforces_horizon_cutoff_and_ranks_universe() -> None:
    result = score()

    assert result.signal_date == date(2026, 6, 19)
    assert result.training_end_date == date(2026, 6, 17)
    assert result.training_date_count == 13
    assert result.training_row_count == 39
    assert result.feature_columns == ("feature_one", "feature_two")
    assert len(result.predictions) == 3
    assert set(result.predictions["ticker"]) == set(TICKERS)
    assert result.predictions["predicted_rank"].tolist() == [1, 2, 3]
    assert result.predictions["score"].notna().all()


def test_completed_market_data_builds_10_day_labels_and_scores() -> None:
    market_data = completed_market_data()
    dates = pd.DatetimeIndex(market_data["date"].drop_duplicates()).sort_values()
    result = score_completed_market_data(
        market_data=market_data,
        expected_tickers=TICKERS,
        horizon_days=10,
        minimum_training_dates=60,
    )

    assert result.signal_date == dates[-1].date()
    assert result.training_end_date == dates[-11].date()
    assert result.training_date_count == 70
    assert len(result.predictions) == 3
    assert np.isfinite(result.predictions["score"]).all()


def test_signal_date_known_target_is_rejected_as_possible_leakage() -> None:
    frame = modeling_dataset()
    latest_date = frame["date"].max()
    frame.loc[frame["date"] == latest_date, TARGET_COLUMN] = 0.5

    with pytest.raises(ValueError, match="possible target leakage"):
        score(frame)


def test_missing_latest_ticker_is_rejected() -> None:
    frame = modeling_dataset()
    latest_date = frame["date"].max()
    frame = frame.loc[
        ~((frame["date"] == latest_date) & (frame["ticker"] == "VCB"))
    ]

    with pytest.raises(ValueError, match="ticker coverage"):
        score(frame)


def test_incomplete_latest_training_target_date_is_rejected() -> None:
    frame = modeling_dataset()
    training_cutoff = sorted(frame["date"].unique())[-3]
    frame.loc[
        (frame["date"] == training_cutoff) & (frame["ticker"] == "FPT"),
        TARGET_COLUMN,
    ] = pd.NA

    with pytest.raises(ValueError, match="complete universe targets"):
        score(frame)


def test_signal_rows_match_ledger_schema_and_have_unique_ids() -> None:
    result = score()
    created_at = datetime(2026, 6, 19, 16, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    rows = build_signal_ledger_rows(
        result=result,
        data_asof_date=result.signal_date,
        intended_execution_date=date(2026, 6, 22),
        created_at=created_at,
    )

    assert len(rows) == 3
    assert len({row["signal_id"] for row in rows}) == 3
    assert set(rows[0]) == set(LEDGER_SCHEMAS["signals.csv"])
    assert all(row["horizon_days"] == 2 for row in rows)
