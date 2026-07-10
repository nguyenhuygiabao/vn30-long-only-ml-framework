from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal

from src.paper_trading.calendar import DateLike, TradingCalendar, normalize_date
from src.paper_trading.schemas import SettlementStatus, Side


ZERO = Decimal("0")


def to_decimal(value: Decimal | int | float | str) -> Decimal:
    if isinstance(value, Decimal):
        return value

    return Decimal(str(value))


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    order_id: str
    execution_date: DateLike
    ticker: str
    side: Side | str
    filled_quantity: int
    execution_price: Decimal | int | float | str
    gross_value: Decimal | int | float | str | None = None
    commission: Decimal | int | float | str = ZERO
    tax: Decimal | int | float | str = ZERO
    slippage: Decimal | int | float | str = ZERO
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.execution_id.strip():
            raise ValueError("execution_id cannot be empty")

        if not self.order_id.strip():
            raise ValueError("order_id cannot be empty")

        ticker = self.ticker.strip().upper()

        if not ticker:
            raise ValueError("ticker cannot be empty")

        side = self.side if isinstance(self.side, Side) else Side(self.side)
        execution_date = normalize_date(self.execution_date)
        execution_price = to_decimal(self.execution_price)

        if self.filled_quantity <= 0:
            raise ValueError("filled_quantity must be positive")

        if execution_price <= ZERO:
            raise ValueError("execution_price must be positive")

        calculated_gross = execution_price * self.filled_quantity
        gross_value = (
            calculated_gross
            if self.gross_value is None
            else to_decimal(self.gross_value)
        )

        if gross_value <= ZERO:
            raise ValueError("gross_value must be positive")

        if gross_value != calculated_gross:
            raise ValueError("gross_value must equal execution_price times quantity")

        commission = to_decimal(self.commission)
        tax = to_decimal(self.tax)
        slippage = to_decimal(self.slippage)

        if any(cost < ZERO for cost in (commission, tax, slippage)):
            raise ValueError("Execution costs cannot be negative")

        if side == Side.SELL and gross_value <= commission + tax + slippage:
            raise ValueError("Sell execution costs cannot exceed gross value")

        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "execution_date", execution_date)
        object.__setattr__(self, "execution_price", execution_price)
        object.__setattr__(self, "gross_value", gross_value)
        object.__setattr__(self, "commission", commission)
        object.__setattr__(self, "tax", tax)
        object.__setattr__(self, "slippage", slippage)

    @property
    def total_costs(self) -> Decimal:
        return self.commission + self.tax + self.slippage

    @property
    def net_cash_effect(self) -> Decimal:
        if self.side == Side.BUY:
            return -(self.gross_value + self.total_costs)

        return self.gross_value - self.total_costs

    def to_row(self) -> dict[str, object]:
        return {
            "execution_id": self.execution_id,
            "order_id": self.order_id,
            "execution_date": self.execution_date.isoformat(),
            "ticker": self.ticker,
            "side": self.side.value,
            "filled_quantity": self.filled_quantity,
            "execution_price": str(self.execution_price),
            "gross_value": str(self.gross_value),
            "commission": str(self.commission),
            "tax": str(self.tax),
            "slippage": str(self.slippage),
            "net_cash_effect": str(self.net_cash_effect),
            "created_at": self.created_at,
        }


@dataclass
class PendingSettlement:
    settlement_id: str
    reference_id: str
    trade_date: date
    settlement_date: date
    ticker: str
    side: Side
    quantity: int
    gross_amount: Decimal
    total_costs: Decimal
    net_cash_effect: Decimal
    status: SettlementStatus = SettlementStatus.PENDING
    settled_at: date | None = None

    def to_row(self) -> dict[str, object]:
        return {
            "settlement_id": self.settlement_id,
            "trade_date": self.trade_date.isoformat(),
            "settlement_date": self.settlement_date.isoformat(),
            "ticker": self.ticker,
            "side": self.side.value,
            "quantity": self.quantity,
            "gross_amount": str(self.gross_amount),
            "fees_and_taxes": str(self.total_costs),
            "status": self.status.value,
            "settled_at": self.settled_at.isoformat() if self.settled_at else "",
            "reference_id": self.reference_id,
        }


def create_pending_settlement(
    execution: ExecutionRecord,
    calendar: TradingCalendar,
    lag_trading_days: int = 2,
) -> PendingSettlement:
    settlement_date = calendar.settlement_date(
        execution.execution_date,
        lag_trading_days=lag_trading_days,
    )

    return PendingSettlement(
        settlement_id=f"settlement-{execution.execution_id}",
        reference_id=execution.execution_id,
        trade_date=execution.execution_date,
        settlement_date=settlement_date,
        ticker=execution.ticker,
        side=execution.side,
        quantity=execution.filled_quantity,
        gross_amount=execution.gross_value,
        total_costs=execution.total_costs,
        net_cash_effect=execution.net_cash_effect,
    )
