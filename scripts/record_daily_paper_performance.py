from __future__ import annotations

import argparse
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.data_loader import load_ohlcv_csv
from src.paper_trading.calendar import normalize_date
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.daily_performance import (
    build_daily_performance_row,
    equal_weight_benchmark_return,
)
from src.paper_trading.settlement import to_decimal
from src.paper_trading.storage import PaperAccountStorage


ZERO = Decimal("0")


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or record daily VN30 paper-account performance."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Performance date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the daily performance row.",
    )

    return parser.parse_args(argv)


def _market_snapshots(
    *,
    raw_ohlcv_path: str,
    performance_date: date,
    multiplier: Decimal,
) -> tuple[
    dict[str, Decimal],
    dict[str, Decimal],
]:
    market_data = load_ohlcv_csv(raw_ohlcv_path)
    market_data["date"] = pd.to_datetime(
        market_data["date"]
    ).dt.date

    available_dates = sorted(
        market_data.loc[
            market_data["date"] <= performance_date,
            "date",
        ].unique()
    )

    if not available_dates:
        raise ValueError(
            "No market data is available on or before "
            f"{performance_date}"
        )

    current_date = available_dates[-1]

    if current_date != performance_date:
        raise ValueError(
            "Market data for the requested performance date "
            f"is unavailable: {performance_date}"
        )

    current_rows = market_data[
        market_data["date"] == current_date
    ].copy()

    current_closes = {
        str(row.ticker).strip().upper(): (
            to_decimal(row.close) * multiplier
        )
        for row in current_rows.itertuples(index=False)
    }

    if len(available_dates) < 2:
        return {}, current_closes

    previous_date = available_dates[-2]
    previous_rows = market_data[
        market_data["date"] == previous_date
    ].copy()

    previous_closes = {
        str(row.ticker).strip().upper(): (
            to_decimal(row.close) * multiplier
        )
        for row in previous_rows.itertuples(index=False)
    }

    return previous_closes, current_closes


def _execution_turnover(
    *,
    executions: pd.DataFrame,
    performance_date: date,
    reference_value: Decimal,
) -> Decimal:
    if executions.empty or reference_value <= ZERO:
        return ZERO

    dated = executions[
        executions["execution_date"]
        == performance_date.isoformat()
    ]

    if dated.empty:
        return ZERO

    gross_value = sum(
        (
            to_decimal(value)
            for value in dated["gross_value"].tolist()
        ),
        start=ZERO,
    )

    return gross_value / reference_value


def _skipped_trade_count(
    *,
    skipped: pd.DataFrame,
    performance_date: date,
) -> int:
    if skipped.empty:
        return 0

    return int(
        (
            skipped["date"]
            == performance_date.isoformat()
        ).sum()
    )


def _replace_performance_date(
    *,
    storage: PaperAccountStorage,
    existing: pd.DataFrame,
    row: dict[str, object],
) -> None:
    target_date = str(row["date"])

    retained = existing[
        existing["date"] != target_date
    ].copy()

    rows = retained.to_dict(orient="records")
    rows.append(row)
    rows.sort(key=lambda item: str(item["date"]))

    storage.replace_rows(
        "daily_performance.csv",
        rows,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    performance_date = normalize_date(args.date)
    config = load_paper_trading_config(args.config)

    storage = PaperAccountStorage(
        config["output"]["directory"]
    )
    storage.validate_all_ledgers()

    broker = storage.load_broker_state()
    performance = storage.read_ledger(
        "daily_performance.csv"
    )
    executions = storage.read_ledger("executions.csv")
    skipped = storage.read_ledger("skipped_trades.csv")

    previous_closes, current_closes = _market_snapshots(
        raw_ohlcv_path=config["model"]["raw_ohlcv_path"],
        performance_date=performance_date,
        multiplier=to_decimal(
            config["market_data"]["price_multiplier_to_vnd"]
        ),
    )

    prior_performance = performance[
        performance["date"]
        < performance_date.isoformat()
    ].sort_values("date")

    if prior_performance.empty:
        benchmark_return = ZERO
        reference_value = (
            broker.settled_cash
            + broker.unsettled_cash
            + sum(
                current_closes.get(ticker, ZERO)
                * position.economic_quantity
                for ticker, position in broker.positions.items()
            )
        )
    else:
        benchmark_return = equal_weight_benchmark_return(
            previous_closes,
            current_closes,
        )
        reference_value = to_decimal(
            prior_performance.iloc[-1][
                "portfolio_value"
            ]
        )

    turnover = _execution_turnover(
        executions=executions,
        performance_date=performance_date,
        reference_value=reference_value,
    )

    skipped_trade_count = _skipped_trade_count(
        skipped=skipped,
        performance_date=performance_date,
    )

    row = build_daily_performance_row(
        performance_date=performance_date,
        broker=broker,
        mark_prices=current_closes,
        previous_rows=prior_performance,
        benchmark_return=benchmark_return,
        turnover=turnover,
        skipped_trade_count=skipped_trade_count,
    )

    print()
    print("VN30 DAILY PAPER PERFORMANCE")
    print("=" * 80)

    for key, value in row.items():
        print(f"{key}: {value}")

    if not args.write:
        print()
        print(
            "Preview only. No account files were changed."
        )
        return 0

    with storage.account_lock():
        storage.validate_all_ledgers()
        current = storage.read_ledger(
            "daily_performance.csv"
        )
        _replace_performance_date(
            storage=storage,
            existing=current,
            row=row,
        )

    print()
    print("DAILY PAPER PERFORMANCE RECORDED SUCCESSFULLY")
    print("Same-date reruns replace the existing row.")
    print("No real broker activity occurred.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
