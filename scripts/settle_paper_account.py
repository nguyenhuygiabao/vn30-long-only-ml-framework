from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.data_loader import load_ohlcv_csv
from src.paper_trading.calendar import normalize_date
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.schemas import SettlementStatus
from src.paper_trading.storage import PaperAccountStorage


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or record due VN30 paper settlements.",
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
    )
    parser.add_argument(
        "--asof-date",
        required=True,
        help="Settlement processing date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist settlement changes to the paper account.",
    )

    return parser.parse_args(argv)


def _latest_mark_prices(
    *,
    raw_ohlcv_path: str,
    asof_date: date,
    multiplier: float,
) -> dict[str, float]:
    market_data = load_ohlcv_csv(raw_ohlcv_path)
    market_data["date"] = pd.to_datetime(
        market_data["date"]
    ).dt.date

    eligible = market_data[
        market_data["date"] <= asof_date
    ].copy()

    if eligible.empty:
        return {}

    latest = (
        eligible.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
    )

    return {
        str(row.ticker).strip().upper(): float(row.close) * multiplier
        for row in latest.itertuples(index=False)
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asof_date = normalize_date(args.asof_date)
    config = load_paper_trading_config(args.config)

    storage = PaperAccountStorage(config["output"]["directory"])
    storage.validate_all_ledgers()
    broker = storage.load_broker_state()

    due = [
        settlement
        for settlement in broker.pending_settlements
        if settlement.status == SettlementStatus.PENDING
        and settlement.settlement_date <= asof_date
    ]

    print()
    print("VN30 PAPER SETTLEMENT")
    print("=" * 80)
    print(f"As-of date: {asof_date}")
    print(f"Mode: {'record' if args.write else 'preview'}")
    print(f"Due settlements: {len(due)}")

    if due:
        preview = pd.DataFrame([
            {
                "settlement_id": settlement.settlement_id,
                "ticker": settlement.ticker,
                "side": settlement.side.value,
                "quantity": settlement.quantity,
                "trade_date": settlement.trade_date,
                "settlement_date": settlement.settlement_date,
                "net_cash_effect": settlement.net_cash_effect,
            }
            for settlement in due
        ])

        print()
        print(preview.to_string(index=False))
    else:
        print("No settlements are due.")

    if not args.write:
        print()
        print("Preview only. No account files were changed.")
        return 0

    mark_prices = _latest_mark_prices(
        raw_ohlcv_path=config["model"]["raw_ohlcv_path"],
        asof_date=asof_date,
        multiplier=float(
            config["market_data"]["price_multiplier_to_vnd"]
        ),
    )

    with storage.account_lock():
        storage.validate_all_ledgers()
        broker = storage.load_broker_state()
        settled = broker.settle_due(asof_date)
        storage.save_broker_state(
            broker,
            mark_prices=mark_prices,
        )

    print()
    print(f"Settlements recorded: {len(settled)}")
    print(f"Settled cash: {broker.settled_cash}")
    print(f"Unsettled cash: {broker.unsettled_cash}")
    print(f"Buying power: {broker.buying_power}")
    print("PAPER SETTLEMENT COMPLETED SUCCESSFULLY")
    print("No real broker activity occurred.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
