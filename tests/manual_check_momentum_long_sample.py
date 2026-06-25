import math

import pandas as pd

from src.features_momentum import add_momentum_features


dates = pd.bdate_range("2024-01-02", periods=65)

validation_data = pd.DataFrame(
    {
        "date": dates,
        "ticker": "TEST",
        "adjusted_close": range(100, 165),
    }
)

features = add_momentum_features(validation_data)

columns_to_count = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "distance_from_20d_high",
    "distance_from_20d_low",
]

print("Valid values:")
print(features[columns_to_count].count())

first_60d_row = features.iloc[60]
first_20d_row = features.iloc[19]

expected_return_60d = 160 / 100 - 1
expected_distance_from_high = 119 / 119 - 1
expected_distance_from_low = 119 / 100 - 1

print("\nManual calculation checks:")
print(
    "First return_60d is correct:",
    math.isclose(
        first_60d_row["return_60d"],
        expected_return_60d,
        rel_tol=1e-12,
    ),
)

print(
    "First distance_from_20d_high is correct:",
    math.isclose(
        first_20d_row["distance_from_20d_high"],
        expected_distance_from_high,
        rel_tol=1e-12,
    ),
)

print(
    "First distance_from_20d_low is correct:",
    math.isclose(
        first_20d_row["distance_from_20d_low"],
        expected_distance_from_low,
        rel_tol=1e-12,
    ),
)
