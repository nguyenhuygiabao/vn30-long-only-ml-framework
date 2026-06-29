import pandas as pd

from src.price_limit import (
    HOSE_NORMAL_DAILY_LIMIT,
    HOSE_WIDE_DAILY_LIMIT,
    add_estimated_price_limits,
    estimate_ceiling_price,
    estimate_floor_price,
    get_hose_tick_size,
    round_down_to_tick,
    round_up_to_tick,
)


print("Normal limit:", HOSE_NORMAL_DAILY_LIMIT)
print("Wide limit:", HOSE_WIDE_DAILY_LIMIT)

print("\nTick sizes:")
print("9,000:", get_hose_tick_size(9_000))
print("25,000:", get_hose_tick_size(25_000))
print("80,000:", get_hose_tick_size(80_000))

print("\nRounding:")
print("Round down 10,735:", round_down_to_tick(10_735))
print("Round up 9,341:", round_up_to_tick(9_341))

print("\nEstimated limits:")
print("Normal ceiling from 10,000:", estimate_ceiling_price(10_000))
print("Normal floor from 10,000:", estimate_floor_price(10_000))
print(
    "Wide ceiling from 10,000:",
    estimate_ceiling_price(10_000, HOSE_WIDE_DAILY_LIMIT),
)
print(
    "Wide floor from 10,000:",
    estimate_floor_price(10_000, HOSE_WIDE_DAILY_LIMIT),
)

data = pd.DataFrame(
    {
        "date": pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        ),
        "ticker": [
            "AAA",
            "AAA",
            "AAA",
            "AAA",
            "AAA",
        ],
        "close": [
            10_000,
            10_700,
            11_400,
            11_000,
            10_200,
        ],
        "high": [
            10_000,
            10_700,
            11_400,
            11_200,
            11_200,
        ],
        "low": [
            10_000,
            10_700,
            11_400,
            11_000,
            10_200,
        ],
    }
)

result = add_estimated_price_limits(data)

print("\nPrice-limit features:")
print(
    result[
        [
            "date",
            "ticker",
            "close",
            "estimated_ceiling_price",
            "estimated_floor_price",
            "distance_to_ceiling",
            "distance_to_floor",
            "hit_ceiling_today",
            "hit_floor_today",
            "consecutive_ceiling_days",
            "consecutive_floor_days",
        ]
    ].round(6).to_string(index=False)
)

ceiling_counts_correct = result["consecutive_ceiling_days"].tolist() == [
    0,
    1,
    2,
    0,
    0,
]

floor_counts_correct = result["consecutive_floor_days"].tolist() == [
    0,
    0,
    0,
    0,
    1,
]

hit_flags_correct = result["hit_ceiling_today"].tolist() == [
    False,
    True,
    True,
    False,
    False,
] and result["hit_floor_today"].tolist() == [
    False,
    False,
    False,
    False,
    True,
]

print("\nCeiling counts correct:", ceiling_counts_correct)
print("Floor counts correct:", floor_counts_correct)
print("Hit flags correct:", hit_flags_correct)
