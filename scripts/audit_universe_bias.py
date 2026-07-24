from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_ohlcv_csv
from src.universe_bias import (
    assert_complete_membership_interval_data,
    build_membership_interval_audit,
    detect_current_only_history_bias,
)
from src.universe_history import (
    snapshots_to_membership_history,
    validate_constituent_snapshot_coverage,
)


DEFAULT_MARKET_PATH = "data/raw/vnstock/vn30_ohlcv.csv"
DEFAULT_SNAPSHOT_PATH = "data/reference/vn30_constituent_snapshots.csv"
DEFAULT_REPORT_PATH = "reports/tables/universe_bias_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit historical VN30 universe and vendor coverage.",
    )
    parser.add_argument("--market-data", default=DEFAULT_MARKET_PATH)
    parser.add_argument("--snapshots", default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--output", default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    market = load_ohlcv_csv(args.market_data)
    snapshots = pd.read_csv(args.snapshots)

    verified_snapshots = validate_constituent_snapshot_coverage(
        snapshots=snapshots,
        start_date=market["date"].min(),
        end_date=market["date"].max(),
        expected_size=30,
        maximum_gap_days=220,
    )
    membership = snapshots_to_membership_history(
        verified_snapshots,
        expected_size=30,
    )

    audit = build_membership_interval_audit(
        market_data=market,
        membership=membership,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    missing_intervals = int(audit["missing_interval"].sum())
    boundary_reviews = int(
        audit["coverage_status"].eq("boundary_review").sum()
    )
    current_only_bias = detect_current_only_history_bias(
        market_data=market,
        membership=membership,
    )

    print("Historical VN30 universe audit")
    print("Snapshot dates:", verified_snapshots["effective_date"].nunique())
    print("Historical tickers:", membership["ticker"].nunique())
    print("Membership intervals:", len(audit))
    print("Missing intervals:", missing_intervals)
    print("Boundary reviews:", boundary_reviews)
    print("Current-only vendor signature:", current_only_bias)
    print("Audit report:", output_path)

    assert_complete_membership_interval_data(audit)

    if current_only_bias:
        raise RuntimeError(
            "Market data contain current members but no former VN30 members"
        )

    print("Bias-control gate: PASS")


if __name__ == "__main__":
    main()