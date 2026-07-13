from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd

from src.feature_pipeline import build_combined_features
from src.labels import (
    add_leave_one_out_equal_weight_relative_labels,
    build_forward_labels,
)
from src.modeling_utils import get_model_feature_columns
from src.tree_models import (
    create_tree_models,
    extract_feature_importance,
    fit_predict_tree_model,
)
from src.walk_forward_split import KEY_COLUMNS, TARGET_COLUMN


HORIZON_TARGET_COLUMNS: dict[int, str] = {
    1: "forward_relative_return_1d",
    5: "forward_relative_return_5d",
    10: "forward_relative_return_10d",
}
RANK_ENSEMBLE_MODEL_NAME = "rank_ensemble"
DEFAULT_RANK_ENSEMBLE_MEMBERS: tuple[str, ...] = (
    "gradient_boosting",
    "random_forest",
)


@dataclass(frozen=True)
class DailyScoringResult:
    signal_date: date
    training_start_date: date
    training_end_date: date
    training_date_count: int
    training_row_count: int
    feature_columns: tuple[str, ...]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def build_daily_modeling_dataset(
    market_data: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    if horizon_days not in HORIZON_TARGET_COLUMNS:
        raise ValueError(f"Unsupported forecast horizon: {horizon_days}")

    features = build_combined_features(market_data)
    stock_labels = build_forward_labels(market_data)
    labels = add_leave_one_out_equal_weight_relative_labels(stock_labels)
    horizon_target = HORIZON_TARGET_COLUMNS[horizon_days]
    selected_labels = labels[KEY_COLUMNS + [horizon_target]].copy()
    selected_labels = selected_labels.rename(
        columns={horizon_target: TARGET_COLUMN}
    )

    return features.merge(
        selected_labels,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )


def _normalized_expected_tickers(
    expected_tickers: Iterable[str],
) -> tuple[str, ...]:
    tickers = tuple(
        sorted({str(ticker).strip().upper() for ticker in expected_tickers})
    )

    if not tickers:
        raise ValueError("Expected ticker universe cannot be empty")

    return tickers


def _rank_ensemble_scores(
    member_scores: dict[str, np.ndarray],
) -> np.ndarray:
    if not member_scores:
        raise ValueError("Rank ensemble requires at least one member model")

    lengths = {len(scores) for scores in member_scores.values()}

    if len(lengths) != 1:
        raise ValueError("Rank ensemble member predictions have inconsistent lengths")

    ranked_scores = []

    for model_name, scores in member_scores.items():
        if not np.isfinite(scores).all():
            raise ValueError(f"Rank ensemble member {model_name} produced non-finite scores")

        ranked_scores.append(
            pd.Series(scores).rank(method="first", ascending=False).to_numpy()
        )

    mean_rank = np.mean(ranked_scores, axis=0)
    return 1.0 - mean_rank / (len(mean_rank) + 1.0)


def score_latest_modeling_dataset(
    modeling_dataset: pd.DataFrame,
    expected_tickers: Iterable[str],
    horizon_days: int,
    model_name: str = "gradient_boosting",
    minimum_training_dates: int = 60,
) -> DailyScoringResult:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    if minimum_training_dates <= 0:
        raise ValueError("minimum_training_dates must be positive")

    required_columns = {"date", "ticker", TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(modeling_dataset.columns))

    if missing_columns:
        raise ValueError(f"Modeling dataset is missing columns: {missing_columns}")

    working = modeling_dataset.copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise").dt.normalize()
    working["ticker"] = working["ticker"].astype(str).str.strip().str.upper()

    duplicate_count = int(working.duplicated(KEY_COLUMNS).sum())

    if duplicate_count:
        raise ValueError(f"Modeling dataset contains {duplicate_count} duplicate keys")

    dates = pd.DatetimeIndex(working["date"].drop_duplicates()).sort_values()

    if len(dates) <= horizon_days:
        raise ValueError("Not enough market dates to enforce the forecast horizon")

    signal_timestamp = dates[-1]
    training_cutoff = dates[-(horizon_days + 1)]
    expected = _normalized_expected_tickers(expected_tickers)
    signal_rows = working.loc[working["date"] == signal_timestamp].copy()
    observed = tuple(sorted(signal_rows["ticker"].unique().tolist()))

    if observed != expected or len(signal_rows) != len(expected):
        missing = sorted(set(expected).difference(observed))
        unexpected = sorted(set(observed).difference(expected))
        raise ValueError(
            "Signal date ticker coverage does not match the universe. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    if signal_rows[TARGET_COLUMN].notna().any():
        raise ValueError(
            "Signal-date targets must be unknown; refusing possible target leakage"
        )

    training_rows = working.loc[
        (working["date"] <= training_cutoff)
        & working[TARGET_COLUMN].notna()
    ].copy()
    cutoff_rows = training_rows.loc[training_rows["date"] == training_cutoff]
    cutoff_tickers = tuple(sorted(cutoff_rows["ticker"].unique().tolist()))

    if cutoff_tickers != expected or len(cutoff_rows) != len(expected):
        raise ValueError(
            "Latest fully realized training date lacks complete universe targets"
        )

    training_dates = pd.DatetimeIndex(
        training_rows["date"].drop_duplicates()
    ).sort_values()

    if len(training_dates) < minimum_training_dates:
        raise ValueError(
            f"Only {len(training_dates)} training dates are available; "
            f"at least {minimum_training_dates} are required"
        )

    if training_rows[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training targets must contain at least two distinct values")

    feature_columns = get_model_feature_columns(training_rows)

    if not feature_columns:
        raise ValueError("No usable model feature columns are available")

    models = create_tree_models()

    if model_name == RANK_ENSEMBLE_MODEL_NAME:
        ensemble_members = DEFAULT_RANK_ENSEMBLE_MEMBERS
    elif model_name in models:
        ensemble_members = (model_name,)
    else:
        raise ValueError(f"Unsupported daily scoring model: {model_name}")

    member_scores: dict[str, np.ndarray] = {}
    importance_frames: list[pd.DataFrame] = []

    for member_name in ensemble_members:
        prediction_series, fitted_pipeline = fit_predict_tree_model(
            model=models[member_name],
            x_train=training_rows[feature_columns],
            y_train=training_rows[TARGET_COLUMN],
            x_test=signal_rows[feature_columns],
        )
        member_scores[member_name] = np.asarray(prediction_series, dtype=float)
        importance = extract_feature_importance(
            fitted_pipeline=fitted_pipeline,
            feature_columns=feature_columns,
            model_name=member_name,
        )
        importance_frames.append(importance)

    scores = (
        _rank_ensemble_scores(member_scores)
        if model_name == RANK_ENSEMBLE_MODEL_NAME
        else member_scores[model_name]
    )

    if not np.isfinite(scores).all():
        raise ValueError("Daily model produced non-finite prediction scores")

    predictions = signal_rows[["date", "ticker"]].copy()
    predictions["model_name"] = model_name
    predictions["horizon_days"] = horizon_days
    predictions["score"] = scores
    predictions = predictions.sort_values(
        ["score", "ticker"],
        ascending=[False, True],
    ).reset_index(drop=True)
    predictions["predicted_rank"] = range(1, len(predictions) + 1)
    predictions = predictions[
        [
            "date",
            "ticker",
            "model_name",
            "horizon_days",
            "score",
            "predicted_rank",
        ]
    ]
    feature_importance = pd.concat(importance_frames, ignore_index=True)

    return DailyScoringResult(
        signal_date=signal_timestamp.date(),
        training_start_date=training_dates.min().date(),
        training_end_date=training_cutoff.date(),
        training_date_count=len(training_dates),
        training_row_count=len(training_rows),
        feature_columns=tuple(feature_columns),
        predictions=predictions,
        feature_importance=feature_importance,
    )


def score_completed_market_data(
    market_data: pd.DataFrame,
    expected_tickers: Iterable[str],
    horizon_days: int,
    model_name: str = "gradient_boosting",
    minimum_training_dates: int = 60,
) -> DailyScoringResult:
    modeling_dataset = build_daily_modeling_dataset(
        market_data=market_data,
        horizon_days=horizon_days,
    )

    return score_latest_modeling_dataset(
        modeling_dataset=modeling_dataset,
        expected_tickers=expected_tickers,
        horizon_days=horizon_days,
        model_name=model_name,
        minimum_training_dates=minimum_training_dates,
    )


def build_signal_ledger_rows(
    result: DailyScoringResult,
    data_asof_date: date,
    intended_execution_date: date,
    created_at: datetime,
) -> list[dict[str, object]]:
    if result.signal_date != data_asof_date:
        raise ValueError("Signal date must match the completed data as-of date")

    rows: list[dict[str, object]] = []

    for prediction in result.predictions.itertuples(index=False):
        signal_id = (
            f"signal-{result.signal_date:%Y%m%d}-"
            f"{prediction.model_name}-{prediction.horizon_days}d-"
            f"{prediction.ticker}"
        )
        rows.append(
            {
                "signal_id": signal_id,
                "data_asof_date": data_asof_date.isoformat(),
                "signal_date": result.signal_date.isoformat(),
                "intended_execution_date": intended_execution_date.isoformat(),
                "ticker": prediction.ticker,
                "model_name": prediction.model_name,
                "horizon_days": prediction.horizon_days,
                "score": str(prediction.score),
                "predicted_rank": prediction.predicted_rank,
                "created_at": created_at.isoformat(),
            }
        )

    return rows
