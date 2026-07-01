from __future__ import annotations

import pandas as pd

from src.risk_model import (
    KEY_COLUMNS,
    MIN_RISK_OBSERVATIONS,
    RETURN_COLUMN,
    RISK_MODEL_PATH,
    RISK_WINDOW,
    build_daily_return_panel,
    load_feature_data,
)


REQUIRED_COLUMNS: list[str] = [
    "date",
    "ticker",
    "return_1d",
    "vn30_return_1d",
    "stock_minus_vn30_return_1d",
    "rolling_covariance_with_vn30",
    "rolling_vn30_variance",
    "rolling_beta_to_vn30",
    "residual_return_1d",
]


def main() -> None:
    features = load_feature_data()
    returns = build_daily_return_panel(features)
    risk_model = pd.read_parquet(RISK_MODEL_PATH)

    returns["date"] = pd.to_datetime(returns["date"])
    risk_model["date"] = pd.to_datetime(risk_model["date"])

    required_columns_present = all(
        column in risk_model.columns
        for column in REQUIRED_COLUMNS
    )

    rows_match_features = len(risk_model) == len(features)
    duplicate_keys_absent = risk_model.duplicated(KEY_COLUMNS).sum() == 0

    check_row = (
        risk_model.dropna(
            subset=[
                "rolling_covariance_with_vn30",
                "rolling_beta_to_vn30",
                "residual_return_1d",
            ]
        )
        .sort_values(KEY_COLUMNS)
        .iloc[0]
    )

    check_ticker = check_row["ticker"]
    check_date = check_row["date"]

    ticker_returns = (
        returns[returns["ticker"] == check_ticker]
        .sort_values("date")
        .reset_index(drop=True)
    )

    check_position = ticker_returns.index[
        ticker_returns["date"] == check_date
    ][0]

    past_window = ticker_returns.iloc[
        max(0, check_position - RISK_WINDOW) : check_position
    ]

    valid_past_window = past_window[
        [
            RETURN_COLUMN,
            "vn30_return_1d",
        ]
    ].dropna()

    manual_covariance = valid_past_window[RETURN_COLUMN].cov(
        valid_past_window["vn30_return_1d"]
    )

    manual_variance = valid_past_window["vn30_return_1d"].var()

    manual_beta = manual_covariance / manual_variance

    manual_residual = (
        check_row[RETURN_COLUMN]
        - manual_beta * check_row["vn30_return_1d"]
    )

    covariance_matches = abs(
        manual_covariance - check_row["rolling_covariance_with_vn30"]
    ) < 1e-12

    variance_matches = abs(
        manual_variance - check_row["rolling_vn30_variance"]
    ) < 1e-12

    beta_matches = abs(
        manual_beta - check_row["rolling_beta_to_vn30"]
    ) < 1e-12

    residual_matches = abs(
        manual_residual - check_row["residual_return_1d"]
    ) < 1e-12

    past_window_excludes_current_date = past_window["date"].max() < check_date

    enough_past_observations = len(valid_past_window) >= MIN_RISK_OBSERVATIONS

    print("Risk model rows:", len(risk_model))
    print("Check ticker:", check_ticker)
    print("Check date:", check_date)
    print("Past window max date:", past_window["date"].max())
    print("Valid past observations:", len(valid_past_window))
    print("Manual covariance:", manual_covariance)
    print("Stored covariance:", check_row["rolling_covariance_with_vn30"])
    print("Manual beta:", manual_beta)
    print("Stored beta:", check_row["rolling_beta_to_vn30"])

    print("\nRequired columns present:", required_columns_present)
    print("Rows match features:", rows_match_features)
    print("Duplicate keys absent:", duplicate_keys_absent)
    print("Past window excludes current date:", past_window_excludes_current_date)
    print("Enough past observations:", enough_past_observations)
    print("Covariance matches manual past-only calculation:", covariance_matches)
    print("Variance matches manual past-only calculation:", variance_matches)
    print("Beta matches manual calculation:", beta_matches)
    print("Residual matches manual calculation:", residual_matches)


if __name__ == "__main__":
    main()