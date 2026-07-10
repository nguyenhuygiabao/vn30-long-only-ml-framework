from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Mapping

from src.paper_trading.calendar import DateLike, TradingCalendar, normalize_date
from src.paper_trading.schemas import SettlementStatus, Side
from src.paper_trading.settlement import (
    ZERO,
    ExecutionRecord,
    PendingSettlement,
    create_pending_settlement,
    to_decimal,
)


@dataclass(frozen=True)
class CashLedgerEntry:
    entry_id: str
    event_date: date
    settlement_date: date
    entry_type: str
    amount: Decimal
    settled_cash_delta: Decimal
    unsettled_cash_delta: Decimal
    reference_id: str
    settled_cash_balance: Decimal
    unsettled_cash_balance: Decimal
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_row(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "event_date": self.event_date.isoformat(),
            "settlement_date": self.settlement_date.isoformat(),
            "entry_type": self.entry_type,
            "amount": str(self.amount),
            "settled_cash_delta": str(self.settled_cash_delta),
            "unsettled_cash_delta": str(self.unsettled_cash_delta),
            "reference_id": self.reference_id,
            "settled_cash_balance": str(self.settled_cash_balance),
            "unsettled_cash_balance": str(self.unsettled_cash_balance),
            "created_at": self.created_at,
        }


@dataclass
class PositionState:
    ticker: str
    issuer_group: str = ""
    settled_shares: int = 0
    unsettled_buy_shares: int = 0
    pending_sell_shares: int = 0
    average_cost: Decimal = ZERO

    @property
    def sellable_quantity(self) -> int:
        return self.settled_shares - self.pending_sell_shares

    @property
    def economic_quantity(self) -> int:
        return (
            self.settled_shares
            + self.unsettled_buy_shares
            - self.pending_sell_shares
        )

    def validate(self) -> None:
        quantities = (
            self.settled_shares,
            self.unsettled_buy_shares,
            self.pending_sell_shares,
            self.sellable_quantity,
            self.economic_quantity,
        )

        if any(quantity < 0 for quantity in quantities):
            raise ValueError(f"Negative position quantity detected for {self.ticker}")

        if self.average_cost < ZERO:
            raise ValueError(f"Negative average cost detected for {self.ticker}")


@dataclass
class PaperBrokerState:
    settled_cash: Decimal | int | float | str
    asof_date: date | None = None
    unsettled_cash: Decimal | int | float | str = ZERO
    positions: dict[str, PositionState] = field(default_factory=dict)
    pending_settlements: list[PendingSettlement] = field(default_factory=list)
    cash_ledger_entries: list[CashLedgerEntry] = field(default_factory=list)
    processed_execution_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.settled_cash = to_decimal(self.settled_cash)
        self.unsettled_cash = to_decimal(self.unsettled_cash)

        if self.asof_date is not None:
            self.asof_date = normalize_date(self.asof_date)

        self.reconcile()

    @classmethod
    def initialize(
        cls,
        initial_cash: Decimal | int | float | str,
        asof_date: DateLike,
    ) -> PaperBrokerState:
        opening_date = normalize_date(asof_date)
        opening_cash = to_decimal(initial_cash)

        if opening_cash <= ZERO:
            raise ValueError("Initial cash must be positive")

        broker = cls(settled_cash=opening_cash, asof_date=opening_date)
        broker.cash_ledger_entries.append(
            CashLedgerEntry(
                entry_id="cash-initial-deposit",
                event_date=opening_date,
                settlement_date=opening_date,
                entry_type="INITIAL_DEPOSIT",
                amount=opening_cash,
                settled_cash_delta=opening_cash,
                unsettled_cash_delta=ZERO,
                reference_id="initial-account-state",
                settled_cash_balance=broker.settled_cash,
                unsettled_cash_balance=broker.unsettled_cash,
            )
        )

        return broker

    @property
    def buying_power(self) -> Decimal:
        return self.settled_cash

    def get_position(self, ticker: str, issuer_group: str = "") -> PositionState:
        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError("ticker cannot be empty")

        if normalized_ticker not in self.positions:
            self.positions[normalized_ticker] = PositionState(
                ticker=normalized_ticker,
                issuer_group=issuer_group,
            )
        elif issuer_group and not self.positions[normalized_ticker].issuer_group:
            self.positions[normalized_ticker].issuer_group = issuer_group

        return self.positions[normalized_ticker]

    def sellable_quantity(self, ticker: str) -> int:
        position = self.positions.get(ticker.strip().upper())

        return position.sellable_quantity if position else 0

    def apply_execution(
        self,
        execution: ExecutionRecord,
        calendar: TradingCalendar,
        issuer_group: str = "",
        settlement_lag_trading_days: int = 2,
    ) -> PendingSettlement:
        if execution.execution_id in self.processed_execution_ids:
            raise ValueError(
                f"Execution has already been processed: {execution.execution_id}"
            )

        settlement = create_pending_settlement(
            execution,
            calendar,
            lag_trading_days=settlement_lag_trading_days,
        )
        position = self.get_position(execution.ticker, issuer_group=issuer_group)

        if execution.side == Side.BUY:
            cash_required = -execution.net_cash_effect

            if cash_required > self.buying_power:
                raise ValueError(
                    f"Insufficient settled cash for {execution.execution_id}: "
                    f"required {cash_required}, available {self.buying_power}"
                )

            previous_quantity = position.economic_quantity
            previous_cost_basis = position.average_cost * previous_quantity
            incoming_cost_basis = execution.gross_value + execution.total_costs

            self.settled_cash -= cash_required
            position.unsettled_buy_shares += execution.filled_quantity
            position.average_cost = (
                previous_cost_basis + incoming_cost_basis
            ) / position.economic_quantity
            settled_cash_delta = execution.net_cash_effect
            unsettled_cash_delta = ZERO
        else:
            if execution.filled_quantity > position.sellable_quantity:
                raise ValueError(
                    f"Insufficient sellable quantity for {execution.execution_id}: "
                    f"required {execution.filled_quantity}, "
                    f"available {position.sellable_quantity}"
                )

            position.pending_sell_shares += execution.filled_quantity
            self.unsettled_cash += execution.net_cash_effect

            if position.economic_quantity == 0:
                position.average_cost = ZERO

            settled_cash_delta = ZERO
            unsettled_cash_delta = execution.net_cash_effect

        self.pending_settlements.append(settlement)
        self.cash_ledger_entries.append(
            CashLedgerEntry(
                entry_id=f"cash-{execution.execution_id}-execution",
                event_date=execution.execution_date,
                settlement_date=settlement.settlement_date,
                entry_type=f"{execution.side.value}_EXECUTION",
                amount=execution.net_cash_effect,
                settled_cash_delta=settled_cash_delta,
                unsettled_cash_delta=unsettled_cash_delta,
                reference_id=execution.execution_id,
                settled_cash_balance=self.settled_cash,
                unsettled_cash_balance=self.unsettled_cash,
            )
        )
        self.processed_execution_ids.add(execution.execution_id)
        self.asof_date = max(
            execution.execution_date,
            self.asof_date or execution.execution_date,
        )
        self.reconcile()

        return settlement

    def settle_due(self, asof_date: DateLike) -> list[PendingSettlement]:
        target_date = normalize_date(asof_date)
        settled: list[PendingSettlement] = []

        pending = sorted(
            self.pending_settlements,
            key=lambda item: (
                item.settlement_date,
                item.trade_date,
                item.settlement_id,
            ),
        )

        for settlement in pending:
            if settlement.status != SettlementStatus.PENDING:
                continue

            if settlement.settlement_date > target_date:
                continue

            position = self.get_position(settlement.ticker)

            if settlement.side == Side.BUY:
                if position.unsettled_buy_shares < settlement.quantity:
                    raise ValueError(
                        f"Cannot settle buy {settlement.settlement_id}: "
                        "unsettled share balance is too low"
                    )

                position.unsettled_buy_shares -= settlement.quantity
                position.settled_shares += settlement.quantity
                settled_cash_delta = ZERO
                unsettled_cash_delta = ZERO
            else:
                if position.pending_sell_shares < settlement.quantity:
                    raise ValueError(
                        f"Cannot settle sell {settlement.settlement_id}: "
                        "pending sell balance is too low"
                    )

                if position.settled_shares < settlement.quantity:
                    raise ValueError(
                        f"Cannot settle sell {settlement.settlement_id}: "
                        "settled share balance is too low"
                    )

                if self.unsettled_cash < settlement.net_cash_effect:
                    raise ValueError(
                        f"Cannot settle sell {settlement.settlement_id}: "
                        "unsettled cash balance is too low"
                    )

                position.pending_sell_shares -= settlement.quantity
                position.settled_shares -= settlement.quantity
                self.unsettled_cash -= settlement.net_cash_effect
                self.settled_cash += settlement.net_cash_effect
                settled_cash_delta = settlement.net_cash_effect
                unsettled_cash_delta = -settlement.net_cash_effect

            settlement.status = SettlementStatus.SETTLED
            settlement.settled_at = target_date
            self.cash_ledger_entries.append(
                CashLedgerEntry(
                    entry_id=f"cash-{settlement.settlement_id}-settlement",
                    event_date=target_date,
                    settlement_date=settlement.settlement_date,
                    entry_type=f"{settlement.side.value}_SETTLEMENT",
                    amount=settlement.net_cash_effect,
                    settled_cash_delta=settled_cash_delta,
                    unsettled_cash_delta=unsettled_cash_delta,
                    reference_id=settlement.settlement_id,
                    settled_cash_balance=self.settled_cash,
                    unsettled_cash_balance=self.unsettled_cash,
                )
            )
            settled.append(settlement)

        self.asof_date = max(target_date, self.asof_date or target_date)
        self.reconcile()

        return settled

    def reconcile(self) -> None:
        if self.settled_cash < ZERO:
            raise ValueError("settled_cash cannot be negative")

        if self.unsettled_cash < ZERO:
            raise ValueError("unsettled_cash cannot be negative")

        for position in self.positions.values():
            position.validate()

        pending_buys: dict[str, int] = {}
        pending_sells: dict[str, int] = {}
        expected_unsettled_cash = ZERO

        for settlement in self.pending_settlements:
            if settlement.status != SettlementStatus.PENDING:
                continue

            if settlement.side == Side.BUY:
                pending_buys[settlement.ticker] = (
                    pending_buys.get(settlement.ticker, 0) + settlement.quantity
                )
            else:
                pending_sells[settlement.ticker] = (
                    pending_sells.get(settlement.ticker, 0) + settlement.quantity
                )
                expected_unsettled_cash += settlement.net_cash_effect

        for ticker, position in self.positions.items():
            if position.unsettled_buy_shares != pending_buys.get(ticker, 0):
                raise ValueError(f"Unsettled buy reconciliation failed for {ticker}")

            if position.pending_sell_shares != pending_sells.get(ticker, 0):
                raise ValueError(f"Pending sell reconciliation failed for {ticker}")

        if self.unsettled_cash != expected_unsettled_cash:
            raise ValueError("Unsettled cash reconciliation failed")

    def cash_snapshot(self) -> dict[str, object]:
        return {
            "asof_date": self.asof_date.isoformat() if self.asof_date else "",
            "settled_cash": str(self.settled_cash),
            "unsettled_cash": str(self.unsettled_cash),
            "buying_power": str(self.buying_power),
        }

    def cash_ledger_rows(self) -> list[dict[str, object]]:
        return [entry.to_row() for entry in self.cash_ledger_entries]

    def settlement_rows(self) -> list[dict[str, object]]:
        return [settlement.to_row() for settlement in self.pending_settlements]

    def position_rows(
        self,
        mark_prices: Mapping[str, Decimal | int | float | str] | None = None,
    ) -> list[dict[str, object]]:
        marks = mark_prices or {}
        total_value = self.settled_cash + self.unsettled_cash
        market_values: dict[str, Decimal] = {}

        for ticker, position in self.positions.items():
            mark_price = to_decimal(marks.get(ticker, ZERO))
            market_value = mark_price * position.economic_quantity
            market_values[ticker] = market_value
            total_value += market_value

        rows: list[dict[str, object]] = []

        for ticker in sorted(self.positions):
            position = self.positions[ticker]
            mark_price = to_decimal(marks.get(ticker, ZERO))
            market_value = market_values[ticker]
            weight = market_value / total_value if total_value > ZERO else ZERO

            rows.append(
                {
                    "asof_date": self.asof_date.isoformat() if self.asof_date else "",
                    "ticker": ticker,
                    "issuer_group": position.issuer_group,
                    "settled_shares": position.settled_shares,
                    "unsettled_buy_shares": position.unsettled_buy_shares,
                    "pending_sell_shares": position.pending_sell_shares,
                    "sellable_quantity": position.sellable_quantity,
                    "average_cost": str(position.average_cost),
                    "mark_price": str(mark_price),
                    "market_value": str(market_value),
                    "weight": str(weight),
                }
            )

        return rows
