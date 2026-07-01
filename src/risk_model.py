from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURES_PATH: str = "data/processed/features_combined.parquet"
RISK_MODEL_PATH: str = "data/processed/risk_model.parquet"

KEY_COLUMNS: list[str] = [
    "date",
    "ticker",
]

RETURN_COLUMN: str = "return_1d"

RISK_WINDOW: int = 60
MIN_RISK_OBSERVATIONS: int = 20

def load_feature_data(path: str = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_daily_return_panel(
    features: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = KEY_COLUMNS + [RETURN_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    returns = features[required_columns].copy()

    returns["date"] = pd.to_datetime(returns["date"])

    returns = returns.sort_values(
        [
            "date",
            "ticker",
        ]
    )

    vn30_daily_return = (
        returns.groupby("date")[RETURN_COLUMN]
        .mean()
        .rename("vn30_return_1d")
        .reset_index()
    )

    returns = returns.merge(
        vn30_daily_return,
        on="date",
        how="left",
    )

    returns["stock_minus_vn30_return_1d"] = (
        returns[RETURN_COLUMN] - returns["vn30_return_1d"]
    )

    return returns

def add_rolling_risk_estimates(
    returns: pd.DataFrame,
    window: int = RISK_WINDOW,
    min_observations: int = MIN_RISK_OBSERVATIONS,
) -> pd.DataFrame:
    risk_frames = []

    for ticker, ticker_returns in returns.groupby("ticker"):
        working = ticker_returns.sort_values("date").copy()

        past_stock_return = working[RETURN_COLUMN].shift(1)
        past_vn30_return = working["vn30_return_1d"].shift(1)

        working["rolling_covariance_with_vn30"] = (
            past_stock_return.rolling(
                window=window,
                min_periods=min_observations,
            ).cov(past_vn30_return)
        )

        working["rolling_vn30_variance"] = (
            past_vn30_return.rolling(
                window=window,
                min_periods=min_observations,
            ).var()
        )

        working["rolling_beta_to_vn30"] = (
            working["rolling_covariance_with_vn30"]
            / working["rolling_vn30_variance"]
        )

        working["residual_return_1d"] = (
            working[RETURN_COLUMN]
            - working["rolling_beta_to_vn30"] * working["vn30_return_1d"]
        )

        risk_frames.append(working)
    
    return (
        pd.concat(
            risk_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )
    

def save_risk_model(
    risk_model: pd.DataFrame,
    output_path: str = RISK_MODEL_PATH,
) -> str:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk_model.to_parquet(
        path,
        index=False,
    )

    return str(path)


def main() -> None:
    features = load_feature_data()
    returns = build_daily_return_panel(features)
    risk_model = add_rolling_risk_estimates(returns)
    output_path = save_risk_model(risk_model)

    print("Risk model path:", output_path)
    print("Risk model rows:", len(risk_model))
    print("Risk model columns:", risk_model.columns.tolist())
    print("Duplicate ticker-date keys:", risk_model.duplicated(KEY_COLUMNS).sum())
    print("Missing stock returns:", risk_model[RETURN_COLUMN].isna().sum())
    print("Missing VN30 returns:", risk_model["vn30_return_1d"].isna().sum())
    print("Missing rolling covariance:", risk_model["rolling_covariance_with_vn30"].isna().sum())
    print("Missing rolling beta:", risk_model["rolling_beta_to_vn30"].isna().sum())
    print("Missing residual returns:", risk_model["residual_return_1d"].isna().sum())

    print("\nSample:")
    print(risk_model.head(10))


if __name__ == "__main__":
    main()