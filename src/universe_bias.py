from __future__ import annotations

import pandas as pd

from src.universe_history import normalize_membership_history


REQUIRED_MARKET_COLUMNS = {"date", "ticker"}


def normalize_market_keys(market_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize market identifiers and reject duplicate ticker-date rows."""
    missing = sorted(REQUIRED_MARKET_COLUMNS.difference(market_data.columns))
    if missing:
        raise ValueError(f"Market data are missing columns: {missing}")

    normalized = market_data.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="raise",
    ).dt.normalize()
    normalized["ticker"] = (
        normalized["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if normalized["ticker"].eq("").any():
        raise ValueError("Market data contain blank tickers")

    duplicates = normalized.duplicated(
        subset=["date", "ticker"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError("Market data contain duplicate ticker-date rows")

    return normalized


def build_membership_interval_audit(
    market_data: pd.DataFrame,
    membership: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    boundary_tolerance_days: int = 10,
) -> pd.DataFrame:
    """Audit vendor observations inside every relevant membership interval."""
    if boundary_tolerance_days < 0:
        raise ValueError("Boundary tolerance cannot be negative")

    market = normalize_market_keys(market_data)
    history = normalize_membership_history(membership)

    if market.empty:
        raise ValueError("Market data are empty")

    market_start = market["date"].min()
    market_end = market["date"].max()

    audit_start = (
        market_start
        if start_date is None
        else pd.Timestamp(start_date).normalize()
    )
    audit_end = (
        market_end
        if end_date is None
        else pd.Timestamp(end_date).normalize()
    )

    if audit_end < audit_start:
        raise ValueError("Audit end date cannot precede start date")

    rows: list[dict[str, object]] = []

    for interval in history.itertuples(index=False):
        interval_start = max(interval.effective_from, audit_start)
        interval_end = (
            audit_end
            if pd.isna(interval.effective_to)
            else min(interval.effective_to, audit_end)
        )

        if interval_end < interval_start:
            continue

        observations = market.loc[
            market["ticker"].eq(interval.ticker)
            & market["date"].between(interval_start, interval_end),
            "date",
        ].sort_values()

        observation_count = len(observations)
        first_observation = (
            observations.iloc[0]
            if observation_count
            else pd.NaT
        )
        last_observation = (
            observations.iloc[-1]
            if observation_count
            else pd.NaT
        )

        unique_dates = observations.drop_duplicates()
        date_gaps = unique_dates.diff().dt.days.dropna()
        maximum_gap_days = (
            int(date_gaps.max())
            if not date_gaps.empty
            else 0
        )

        missing_interval = observation_count == 0
        delayed_start = (
            not missing_interval
            and first_observation
            > interval_start + pd.Timedelta(days=boundary_tolerance_days)
        )
        early_end = (
            not missing_interval
            and last_observation
            < interval_end - pd.Timedelta(days=boundary_tolerance_days)
        )

        if missing_interval:
            status = "missing_interval"
        elif delayed_start or early_end:
            status = "boundary_review"
        else:
            status = "covered"

        rows.append(
            {
                "ticker": interval.ticker,
                "membership_from": interval.effective_from,
                "membership_to": interval.effective_to,
                "audit_from": interval_start,
                "audit_to": interval_end,
                "market_observations": observation_count,
                "first_market_observation": first_observation,
                "last_market_observation": last_observation,
                "maximum_calendar_gap_days": maximum_gap_days,
                "missing_interval": missing_interval,
                "delayed_start": delayed_start,
                "early_end": early_end,
                "coverage_status": status,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["ticker", "membership_from"],
        kind="stable",
    ).reset_index(drop=True)


def assert_complete_membership_interval_data(
    audit: pd.DataFrame,
) -> None:
    """Fail if any historical membership interval has no vendor data."""
    required = {
        "ticker",
        "audit_from",
        "audit_to",
        "missing_interval",
    }
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise ValueError(f"Universe audit is missing columns: {missing}")

    failures = audit.loc[audit["missing_interval"]].copy()
    if failures.empty:
        return

    details = [
        (
            f"{row.ticker}:"
            f"{pd.Timestamp(row.audit_from).date()}"
            f"..{pd.Timestamp(row.audit_to).date()}"
        )
        for row in failures.itertuples(index=False)
    ]
    raise ValueError(
        "Historical members have membership intervals with no market data: "
        + ", ".join(details)
    )


def detect_current_only_history_bias(
    market_data: pd.DataFrame,
    membership: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
) -> bool:
    """Detect a vendor dataset containing current members but no former members."""
    market = normalize_market_keys(market_data)
    history = normalize_membership_history(membership)

    as_of = (
        market["date"].max()
        if as_of_date is None
        else pd.Timestamp(as_of_date).normalize()
    )

    known_by_date = history.loc[
        history["effective_from"] <= as_of
    ]
    current = known_by_date.loc[
        known_by_date["effective_to"].isna()
        | (known_by_date["effective_to"] >= as_of),
        "ticker",
    ]
    historical = set(known_by_date["ticker"])
    former = historical.difference(set(current))
    observed = set(
        market.loc[market["date"] <= as_of, "ticker"]
    )

    return bool(former) and observed.isdisjoint(former)