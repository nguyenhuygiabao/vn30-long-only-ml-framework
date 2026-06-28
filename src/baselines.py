from __future__ import annotations

import pandas as pd

from src.walk_forward_split import (
    TARGET_COLUMN,
    build_modeling_dataset,
    load_modeling_inputs,
    separate_target_rows,
)

BASELINE_SCORE_COLUMNS: list[str] = [
    "score_equal",
    "score_return_1d",
    "score_return_5d",
    "score_return_10d",
]

FORBIDDEN_SCORE_COLUMNS: list[str] = [
    "date",
    "ticker",
    TARGET_COLUMN,
]

def validate_score_columns(score_columns: list[str]) -> None:
    forbidden_used = [
        column
        for column in score_columns
        if column in FORBIDDEN_SCORE_COLUMNS
    ]

    if forbidden_used:
        raise ValueError(
            f"Score columns cannot contain identifiers or target columns: {forbidden_used}"
        )

def load_baseline_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, labels = load_modeling_inputs()

    modeling_dataset = build_modeling_dataset(
        features=features,
        labels=labels,
    )

    historical_rows, prediction_rows = separate_target_rows(
        modeling_dataset=modeling_dataset,
    )

    return modeling_dataset, historical_rows, prediction_rows

def add_equal_score(data: pd.DataFrame) -> pd.DataFrame:
    scored_data = data.copy()

    scored_data["score_equal"] = 0.0

    return scored_data

def add_momentum_scores(data: pd.DataFrame) -> pd.DataFrame:
    scored_data = data.copy()

    scored_data["score_return_1d"] = scored_data["return_1d"]
    scored_data["score_return_5d"] = scored_data["return_5d"]
    scored_data["score_return_10d"] = scored_data["return_10d"]

    return scored_data

def rank_scores_by_date(
        data: pd.DataFrame,
        score_columns: list[str],
) -> pd.DataFrame:
    validate_score_columns(score_columns)

    ranked_data = data.copy()

    for score_column in score_columns:
        rank_column = score_column.replace("score_","rank_")

        ranked_data[rank_column] = (
            ranked_data.groupby("date")[score_column].rank(
                ascending = False,
                method = "first"
            )
        )
    return ranked_data

def evaluate_rank_ic_by_date(
    data: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    validate_score_columns([score_column])

    required_columns = [
        "date",
        score_column,
        TARGET_COLUMN,
    ]

    evaluation_data = data[required_columns].dropna().copy()

    rank_ic_rows = []

    for date, date_data in evaluation_data.groupby("date"):
        if len(date_data) < 2:
            continue

        if date_data[score_column].nunique() < 2:
            continue

        if date_data[TARGET_COLUMN].nunique() < 2:
            continue

        rank_ic = date_data[score_column].corr(
            date_data[TARGET_COLUMN],
            method="spearman",
        )

        rank_ic_rows.append(
            {
                "date": date,
                "score_column": score_column,
                "rank_ic": rank_ic,
            }
        )

    return pd.DataFrame(rank_ic_rows)

def summarize_rank_ic(
        data: pd.DataFrame,
        score_columns: list[str],
) -> pd.DataFrame: 
        summary_rows = []

        for score_column in score_columns:
            rank_ic_by_date = evaluate_rank_ic_by_date(
                data = data,
                score_column = score_column,
            )
        
            evaluated_dates = len(rank_ic_by_date)

            if evaluated_dates == 0:
                average_rank_ic = None
            else:
                average_rank_ic = rank_ic_by_date["rank_ic"].mean()

            summary_rows.append(
                {
                    "score_column": score_column,
                    "evaluated_dates": evaluated_dates,
                    "average_rank_ic": average_rank_ic,
                }
        ) 
        return pd.DataFrame(summary_rows)

def main() -> None:
    modeling_dataset, historical_rows, prediction_rows = load_baseline_dataset()

    historical_rows = add_equal_score(historical_rows)
    prediction_rows = add_equal_score(prediction_rows)

    historical_rows = add_momentum_scores(historical_rows)
    prediction_rows = add_momentum_scores(prediction_rows)

    historical_rows = rank_scores_by_date(
        data = historical_rows,
        score_columns= BASELINE_SCORE_COLUMNS,
    )
    prediction_rows = rank_scores_by_date(
        data = prediction_rows,
        score_columns= BASELINE_SCORE_COLUMNS,
    )

    print("Modeling rows:", len(modeling_dataset))
    print("Modeling columns:", len(modeling_dataset.columns))
    print("Historical rows:", len(historical_rows))
    print("Prediction-only rows:", len(prediction_rows))
    print("Target column:", TARGET_COLUMN)
    print("Historical targets all known:", historical_rows[TARGET_COLUMN].notna().all())
    print("Prediction targets all missing:", prediction_rows[TARGET_COLUMN].isna().all())

    print("Equal score column exists", "score_equal" in historical_rows.columns)
    print("Unique equal scores:", historical_rows["score_equal"].unique().tolist())

    print("1-day momentum score exists:", "score_return_1d" in historical_rows.columns)
    print("5-day momentum score exists:", "score_return_5d" in historical_rows.columns)
    print("10-day momentum score exists:", "score_return_10d" in historical_rows.columns)
    print("Missing 1-day momentum scores:", historical_rows["score_return_1d"].isna().sum())
    print("Missing 5-day momentum scores:", historical_rows["score_return_5d"].isna().sum())
    print("Missing 10-day momentum scores:", historical_rows["score_return_10d"].isna().sum())
    print("Equal rank column exists:", "rank_equal" in historical_rows.columns)
    print("1-day momentum rank column exists:", "rank_return_1d" in historical_rows.columns)
    print("5-day momentum rank column exists:", "rank_return_5d" in historical_rows.columns)
    print("10-day momentum rank column exists:", "rank_return_10d" in historical_rows.columns)
    print("Missing 1-day momentum ranks:", historical_rows["rank_return_1d"].isna().sum())
    print("Missing 5-day momentum ranks:", historical_rows["rank_return_5d"].isna().sum())
    print("Missing 10-day momentum ranks:", historical_rows["rank_return_10d"].isna().sum())
    one_day_rank_ic = evaluate_rank_ic_by_date(
        data=historical_rows,
        score_column="score_return_1d",
    )

    print("1-day momentum Rank IC rows:", len(one_day_rank_ic))

    if len(one_day_rank_ic) > 0:
        print("1-day momentum average Rank IC:", one_day_rank_ic["rank_ic"].mean())

    rank_ic_summary = summarize_rank_ic(
        data = historical_rows, 
        score_columns = BASELINE_SCORE_COLUMNS,
    )

    print("\nRank IC summary:")
    print(rank_ic_summary)

if __name__ == "__main__":
    main()
