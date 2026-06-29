import pandas as pd

from src.linear_models import (
    calculate_rank_ic_by_date,
    calculate_top_n_average_return,
    create_linear_models,
    get_model_feature_columns,
    predict_models_for_windows,
    summarize_model_performance,
)
from src.walk_forward_split import TARGET_COLUMN


synthetic_rows = []

for date in [
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2024-01-02"),
    pd.Timestamp("2024-01-03"),
    pd.Timestamp("2024-01-04"),
    pd.Timestamp("2024-01-05"),
]:
    for ticker_number in range(1, 4):
        ticker = f"T{ticker_number:02d}"

        synthetic_rows.append(
            {
                "date": date,
                "ticker": ticker,
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 101.0,
                "adjusted_close": 101.0,
                "volume": 1000000,
                "value_traded": 101000000,
                "return_1d": ticker_number / 100,
                "return_3d": ticker_number / 90,
                "distance_from_20d_high": -ticker_number / 100,
                "positive_shock_1d": 0,
                "forward_return_5d": 0.02 + ticker_number / 1000,
                "vn30_forward_return_5d": 0.02,
                TARGET_COLUMN: ticker_number / 1000,
            }
        )

synthetic_data = pd.DataFrame(synthetic_rows)

feature_columns = get_model_feature_columns(synthetic_data)

print("Synthetic rows:", len(synthetic_data))
print("Feature columns:", feature_columns)

windows = [
    {
        "train_dates": pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ]
        ),
        "validation_dates": pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-03"),
            ]
        ),
        "test_dates": pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-04"),
            ]
        ),
    },
    {
        "train_dates": pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ]
        ),
        "validation_dates": pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-04"),
            ]
        ),
        "test_dates": pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-05"),
            ]
        ),
    },
]

models = create_linear_models()

predictions = predict_models_for_windows(
    models=models,
    historical_rows=synthetic_data,
    windows=windows,
    feature_columns=feature_columns,
)

required_prediction_columns = [
    "date",
    "ticker",
    "predicted_return",
    "actual_return",
    "model_name",
]

prediction_columns_correct = predictions.columns.tolist() == required_prediction_columns
prediction_rows_correct = len(predictions) == 12
models_correct = sorted(predictions["model_name"].unique().tolist()) == [
    "elastic_net",
    "ridge",
]

print("\nPredictions:")
print(predictions.round(6))

print("\nPrediction columns correct:", prediction_columns_correct)
print("Prediction rows correct:", prediction_rows_correct)
print("Models correct:", models_correct)

rank_ic_by_date = calculate_rank_ic_by_date(
    predictions=predictions,
)

top_n_returns = calculate_top_n_average_return(
    predictions=predictions,
    top_n=2,
)

performance_summary = summarize_model_performance(
    predictions=predictions,
    top_n=2,
)

rank_ic_rows_correct = len(rank_ic_by_date) == 4
top_n_rows_correct = len(top_n_returns) == 4
summary_rows_correct = len(performance_summary) == 2

print("\nRank IC by date:")
print(rank_ic_by_date.round(6))

print("\nTop-N returns:")
print(top_n_returns.round(6))

print("\nPerformance summary:")
print(performance_summary.round(6))

print("\nRank IC rows correct:", rank_ic_rows_correct)
print("Top-N rows correct:", top_n_rows_correct)
print("Summary rows correct:", summary_rows_correct)
