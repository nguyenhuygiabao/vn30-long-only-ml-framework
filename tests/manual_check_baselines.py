import pandas as pd

from src.baselines import (
    BASELINE_SCORE_COLUMNS,
    add_equal_score,
    add_momentum_scores,
    evaluate_rank_ic_by_date,
    rank_scores_by_date,
    summarize_rank_ic,
)
from src.walk_forward_split import TARGET_COLUMN


synthetic_data = pd.DataFrame(
    [
        {
            "date": pd.Timestamp("2024-01-02"),
            "ticker": "AAA",
            "return_1d": 0.03,
            "return_5d": 0.05,
            "return_10d": 0.08,
            TARGET_COLUMN: 0.06,
        },
        {
            "date": pd.Timestamp("2024-01-02"),
            "ticker": "BBB",
            "return_1d": 0.02,
            "return_5d": 0.04,
            "return_10d": 0.07,
            TARGET_COLUMN: 0.03,
        },
        {
            "date": pd.Timestamp("2024-01-02"),
            "ticker": "CCC",
            "return_1d": 0.01,
            "return_5d": 0.03,
            "return_10d": 0.06,
            TARGET_COLUMN: -0.01,
        },
    ]
)

scored_data = add_equal_score(synthetic_data)
scored_data = add_momentum_scores(scored_data)

ranked_data = rank_scores_by_date(
    data=scored_data,
    score_columns=BASELINE_SCORE_COLUMNS,
)

one_day_rank_ic = evaluate_rank_ic_by_date(
    data=ranked_data,
    score_column="score_return_1d",
)

rank_ic_summary = summarize_rank_ic(
    data=ranked_data,
    score_columns=BASELINE_SCORE_COLUMNS,
)

print("Synthetic rows:", len(ranked_data))
print("Score columns exist:", all(column in ranked_data.columns for column in BASELINE_SCORE_COLUMNS))
print("rank_return_1d exists:", "rank_return_1d" in ranked_data.columns)

print("\nRanked synthetic data:")
print(
    ranked_data[
        [
            "date",
            "ticker",
            "score_return_1d",
            "rank_return_1d",
            TARGET_COLUMN,
        ]
    ]
)

print("\n1-day Rank IC rows:", len(one_day_rank_ic))

if len(one_day_rank_ic) > 0:
    print("1-day Rank IC value:", one_day_rank_ic["rank_ic"].iloc[0])

print("\nRank IC summary:")
print(rank_ic_summary)

try:
    rank_scores_by_date(
        data=scored_data,
        score_columns=[
            TARGET_COLUMN,
        ],
    )

    target_leakage_guard_valid = False

except ValueError:
    target_leakage_guard_valid = True

print("\nTarget leakage guard works:", target_leakage_guard_valid)