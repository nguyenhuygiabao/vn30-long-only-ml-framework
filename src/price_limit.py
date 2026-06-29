from __future__ import annotations

import pandas as pd


HOSE_NORMAL_DAILY_LIMIT: float = 0.07
HOSE_WIDE_DAILY_LIMIT: float = 0.20


def get_hose_tick_size(
    price: float,
) -> float:
    if pd.isna(price):
        return float("nan")

    if price < 10_000:
        return 10.0

    if price < 50_000:
        return 50.0

    return 100.0


def round_down_to_tick(
    price: float,
) -> float:
    if pd.isna(price):
        return float("nan")

    tick_size = get_hose_tick_size(price)

    return price // tick_size * tick_size


def round_up_to_tick(
    price: float,
) -> float:
    if pd.isna(price):
        return float("nan")

    tick_size = get_hose_tick_size(price)

    return -(-price // tick_size) * tick_size


def estimate_ceiling_price(
    reference_price: float,
    daily_limit_rate: float = HOSE_NORMAL_DAILY_LIMIT,
) -> float:
    if pd.isna(reference_price):
        return float("nan")

    raw_ceiling_price = reference_price * (1 + daily_limit_rate)

    return round_down_to_tick(raw_ceiling_price)


def estimate_floor_price(
    reference_price: float,
    daily_limit_rate: float = HOSE_NORMAL_DAILY_LIMIT,
) -> float:
    if pd.isna(reference_price):
        return float("nan")

    raw_floor_price = reference_price * (1 - daily_limit_rate)

    return round_up_to_tick(raw_floor_price)


def count_consecutive_true(
    values: pd.Series,
) -> pd.Series:
    counts = []
    current_count = 0

    for value in values:
        if bool(value):
            current_count += 1
        else:
            current_count = 0

        counts.append(current_count)

    return pd.Series(
        counts,
        index=values.index,
    )


def add_estimated_price_limits(
    data: pd.DataFrame,
    price_column: str = "close",
    daily_limit_rate: float = HOSE_NORMAL_DAILY_LIMIT,
) -> pd.DataFrame:
    working = data.copy()

    working = working.sort_values(
        [
            "ticker",
            "date",
        ]
    )

    working["reference_price"] = working.groupby("ticker")[price_column].shift(1)

    working["estimated_ceiling_price"] = working["reference_price"].apply(
        lambda price: estimate_ceiling_price(
            reference_price=price,
            daily_limit_rate=daily_limit_rate,
        )
    )

    working["estimated_floor_price"] = working["reference_price"].apply(
        lambda price: estimate_floor_price(
            reference_price=price,
            daily_limit_rate=daily_limit_rate,
        )
    )

    working["distance_to_ceiling"] = (
        working["estimated_ceiling_price"] - working[price_column])/working[price_column]

    working["distance_to_floor"] = (
        working[price_column] - working["estimated_floor_price"])/working[price_column]

    ceiling_check_column = "high" if "high" in working.columns else price_column
    floor_check_column = "low" if "low" in working.columns else price_column

    working["hit_ceiling_today"] = (
        working[ceiling_check_column] >= working["estimated_ceiling_price"]
    )

    working["hit_floor_today"] = (
        working[floor_check_column] <= working["estimated_floor_price"]
    )

    working["consecutive_ceiling_days"] = working.groupby("ticker")[
        "hit_ceiling_today"
    ].transform(count_consecutive_true)

    working["consecutive_floor_days"] = working.groupby("ticker")[
        "hit_floor_today"
    ].transform(count_consecutive_true)

    return working.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(drop=True)
