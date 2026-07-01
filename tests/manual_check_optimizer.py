from __future__ import annotations

import pandas as pd

from src.optimizer import (
    ISSUER_GROUP_COLUMN,
    KEY_COLUMNS,
    MAX_ISSUER_GROUP_WEIGHT,
    MAX_TURNOVER,
    MAX_WEIGHT,
    MIN_WEIGHT,
    MODEL_NAME,
    OPTIMIZED_WEIGHTS_PATH,
    TREE_MODEL_PREDICTIONS_PATH,
    calculate_turnover,
)

TOLERANCE: float = 1e-8

REQUIRED_COLUMNS: list[str] = [
    "date",
    "ticker",
    "weight",
    "predicted_return",
    "actual_return",
    "model_name",
    "portfolio_turnover",
    "issuer_group",
]


def main() -> None:
    weights = pd.read_parquet(OPTIMIZED_WEIGHTS_PATH)
    predictions = pd.read_parquet(TREE_MODEL_PREDICTIONS_PATH)

    weights["date"] = pd.to_datetime(weights["date"])
    predictions["date"] = pd.to_datetime(predictions["date"])

    model_predictions = predictions[
        predictions["model_name"] == MODEL_NAME
    ]

    required_columns_present = all(
        column in weights.columns
        for column in REQUIRED_COLUMNS
    )

    duplicate_keys_absent = weights.duplicated(KEY_COLUMNS).sum() == 0

    weights_non_negative = weights["weight"].min() >= 0

    tiny_weights_removed = weights["weight"].min() >= MIN_WEIGHT

    max_weight_respected = weights["weight"].max() <= MAX_WEIGHT + TOLERANCE

    daily_weight_sum = weights.groupby("date")["weight"].sum()

    daily_issuer_group_weight = weights.groupby(
        [
            "date",
            ISSUER_GROUP_COLUMN,
        ]
    )["weight"].sum()

    issuer_group_cap_respected = (
        daily_issuer_group_weight.max()
        <= MAX_ISSUER_GROUP_WEIGHT + TOLERANCE
    )

    issuer_group_complete = weights[ISSUER_GROUP_COLUMN].notna().all()

    weight_sum_respected = daily_weight_sum.max() <= 1.0 + TOLERANCE

    stored_turnover_respected = (
        weights.groupby("date")["portfolio_turnover"].max().max()
        <= MAX_TURNOVER + TOLERANCE
    )

    recalculated_turnovers = []

    previous_weights = pd.Series(dtype=float)

    for date, date_weights in weights.groupby("date"):
        current_weights = date_weights.set_index("ticker")["weight"]

        turnover = calculate_turnover(
            old_weights=previous_weights,
            new_weights=current_weights,
        )

        recalculated_turnovers.append(
            {
                "date": date,
                "recalculated_turnover": turnover,
                "stored_turnover": date_weights["portfolio_turnover"].max(),
            }
        )

        previous_weights = current_weights

    turnover_check = pd.DataFrame(recalculated_turnovers)

    recalculated_turnover_respected = (
        turnover_check["recalculated_turnover"].max()
        <= MAX_TURNOVER + TOLERANCE
    )

    stored_turnover_matches_recalculated = (
        (
            turnover_check["recalculated_turnover"]
            - turnover_check["stored_turnover"]
        )
        .abs()
        .max()
        <= TOLERANCE
    )

    dates_match_model_predictions = (
        weights["date"].nunique()
        == model_predictions["date"].nunique()
    )

    only_expected_model = weights["model_name"].eq(MODEL_NAME).all()

    print("Optimized weight rows:", len(weights))
    print("Date count:", weights["date"].nunique())
    print("Ticker count:", weights["ticker"].nunique())
    print("Minimum weight:", weights["weight"].min())
    print("Maximum weight:", weights["weight"].max())
    print("Maximum daily weight sum:", daily_weight_sum.max())
    print("Maximum issuer group weight:", daily_issuer_group_weight.max())
    print("Maximum stored turnover:", weights.groupby("date")["portfolio_turnover"].max().max())
    print("Maximum recalculated turnover:", turnover_check["recalculated_turnover"].max())
    print("Maximum turnover difference:", (
        turnover_check["recalculated_turnover"]
        - turnover_check["stored_turnover"]
    ).abs().max())

    print("\nRequired columns present:", required_columns_present)
    print("Duplicate keys absent:", duplicate_keys_absent)
    print("Weights non-negative:", weights_non_negative)
    print("Tiny weights removed:", tiny_weights_removed)
    print("Max weight respected:", max_weight_respected)
    print("Weight sum respected:", weight_sum_respected)
    print("Stored turnover respected:", stored_turnover_respected)
    print("Recalculated turnover respected:", recalculated_turnover_respected)
    print("Stored turnover matches recalculated:", stored_turnover_matches_recalculated)
    print("Dates match model predictions:", dates_match_model_predictions)
    print("Only expected model:", only_expected_model)
    print("Issuer group complete:", issuer_group_complete)
    print("Issuer group cap respected:", issuer_group_cap_respected)

if __name__ == "__main__":
    main()