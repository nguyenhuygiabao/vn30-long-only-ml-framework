from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.paper_trading.broker_state import PaperBrokerState
from src.paper_trading.calendar import TradingCalendar
from src.paper_trading.schemas import LEDGER_SCHEMAS, SettlementStatus, Side
from src.paper_trading.settlement import ExecutionRecord


def build_calendar() -> TradingCalendar:
    return TradingCalendar.from_weekdays("2026-07-01", "2026-07-31")


def build_buy(
    execution_id: str = "execution-buy-1",
    execution_date: str = "2026-07-06",
    quantity: int = 100,
    price: str = "10000",
) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=execution_id,
        order_id=f"order-{execution_id}",
        execution_date=execution_date,
        ticker="FPT",
        side=Side.BUY,
        filled_quantity=quantity,
        execution_price=price,
        commission="1000",
        slippage="1000",
    )


def test_buy_cash_and_shares_remain_unsettled_until_t_plus_two() -> None:
    broker = PaperBrokerState(settled_cash="10000000")
    calendar = build_calendar()
    settlement = broker.apply_execution(
        build_buy(),
        calendar,
        issuer_group="FPT",
    )
    position = broker.positions["FPT"]

    assert settlement.settlement_date == date(2026, 7, 8)
    assert broker.settled_cash == Decimal("8998000")
    assert broker.buying_power == Decimal("8998000")
    assert broker.unsettled_cash == Decimal("0")
    assert position.settled_shares == 0
    assert position.unsettled_buy_shares == 100
    assert position.sellable_quantity == 0

    assert broker.settle_due("2026-07-07") == []
    settled = broker.settle_due("2026-07-08")

    assert settled == [settlement]
    assert settlement.status == SettlementStatus.SETTLED
    assert position.settled_shares == 100
    assert position.unsettled_buy_shares == 0
    assert position.sellable_quantity == 100


def test_initial_account_state_records_cash_ledger_entry() -> None:
    broker = PaperBrokerState.initialize("10000000", "2026-07-06")
    row = broker.cash_ledger_rows()[0]

    assert row["entry_type"] == "INITIAL_DEPOSIT"
    assert row["settled_cash_balance"] == "10000000"
    assert list(row) == list(LEDGER_SCHEMAS["cash_ledger.csv"])


def test_unsettled_bought_shares_cannot_be_sold() -> None:
    broker = PaperBrokerState(settled_cash="10000000")
    calendar = build_calendar()
    broker.apply_execution(build_buy(), calendar)
    sell = ExecutionRecord(
        execution_id="execution-sell-too-early",
        order_id="order-sell-too-early",
        execution_date="2026-07-07",
        ticker="FPT",
        side=Side.SELL,
        filled_quantity=1,
        execution_price="11000",
    )

    with pytest.raises(ValueError, match="Insufficient sellable quantity"):
        broker.apply_execution(sell, calendar)


def test_sell_proceeds_become_buying_power_only_after_settlement() -> None:
    broker = PaperBrokerState(settled_cash="10000000")
    calendar = build_calendar()
    broker.apply_execution(build_buy(), calendar)
    broker.settle_due("2026-07-08")
    sell = ExecutionRecord(
        execution_id="execution-sell-1",
        order_id="order-sell-1",
        execution_date="2026-07-08",
        ticker="FPT",
        side=Side.SELL,
        filled_quantity=40,
        execution_price="12000",
        commission="480",
        tax="480",
        slippage="480",
    )
    settlement = broker.apply_execution(sell, calendar)
    position = broker.positions["FPT"]

    assert settlement.settlement_date == date(2026, 7, 10)
    assert broker.unsettled_cash == Decimal("478560")
    assert broker.buying_power == Decimal("8998000")
    assert position.pending_sell_shares == 40
    assert position.sellable_quantity == 60
    assert list(broker.cash_ledger_rows()[-1]) == list(
        LEDGER_SCHEMAS["cash_ledger.csv"]
    )
    assert list(broker.settlement_rows()[-1]) == list(
        LEDGER_SCHEMAS["settlement_ledger.csv"]
    )
    assert position.economic_quantity == 60

    broker.settle_due("2026-07-10")

    assert broker.unsettled_cash == Decimal("0")
    assert broker.settled_cash == Decimal("9476560")
    assert broker.buying_power == Decimal("9476560")
    assert position.settled_shares == 60
    assert position.pending_sell_shares == 0
    assert position.sellable_quantity == 60


def test_unsettled_sale_proceeds_cannot_fund_new_buy() -> None:
    broker = PaperBrokerState(settled_cash="10000000")
    calendar = build_calendar()
    broker.apply_execution(build_buy(), calendar)
    broker.settle_due("2026-07-08")
    sell = ExecutionRecord(
        execution_id="execution-sell-2",
        order_id="order-sell-2",
        execution_date="2026-07-08",
        ticker="FPT",
        side=Side.SELL,
        filled_quantity=100,
        execution_price="12000",
    )
    broker.apply_execution(sell, calendar)
    oversized_buy = ExecutionRecord(
        execution_id="execution-buy-oversized",
        order_id="order-buy-oversized",
        execution_date="2026-07-09",
        ticker="VNM",
        side=Side.BUY,
        filled_quantity=1000,
        execution_price="10000",
    )

    assert broker.unsettled_cash == Decimal("1200000")

    with pytest.raises(ValueError, match="Insufficient settled cash"):
        broker.apply_execution(oversized_buy, calendar)


def test_duplicate_execution_is_rejected() -> None:
    broker = PaperBrokerState(settled_cash="10000000")
    calendar = build_calendar()
    execution = build_buy()
    broker.apply_execution(execution, calendar)

    with pytest.raises(ValueError, match="already been processed"):
        broker.apply_execution(execution, calendar)


def test_position_snapshot_counts_economic_quantity_once() -> None:
    broker = PaperBrokerState(settled_cash="10000000")
    calendar = build_calendar()
    broker.apply_execution(build_buy(), calendar, issuer_group="FPT")

    rows = broker.position_rows({"FPT": "11000"})

    assert len(rows) == 1
    assert rows[0]["market_value"] == "1100000"
    assert rows[0]["unsettled_buy_shares"] == 100
    assert rows[0]["sellable_quantity"] == 0


def test_reconciliation_detects_corrupted_pending_share_balance() -> None:
    broker = PaperBrokerState(settled_cash="10000000")
    calendar = build_calendar()
    broker.apply_execution(build_buy(), calendar)
    broker.positions["FPT"].unsettled_buy_shares = 99

    with pytest.raises(ValueError, match="Unsettled buy reconciliation failed"):
        broker.reconcile()


def test_execution_gross_value_must_reconcile() -> None:
    with pytest.raises(ValueError, match="gross_value must equal"):
        ExecutionRecord(
            execution_id="execution-bad-gross",
            order_id="order-bad-gross",
            execution_date="2026-07-06",
            ticker="FPT",
            side=Side.BUY,
            filled_quantity=100,
            execution_price="10000",
            gross_value="999999",
        )


def test_execution_row_matches_ledger_schema() -> None:
    row = build_buy().to_row()

    assert list(row) == list(LEDGER_SCHEMAS["executions.csv"])
