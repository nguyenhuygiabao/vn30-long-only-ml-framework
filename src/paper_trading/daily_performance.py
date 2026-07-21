from __future__ import annotations

from decimal import Decimal
from typing import Mapping

import pandas as pd

from src.paper_trading.settlement import to_decimal


ZERO = Decimal("0")
ONE = Decimal("1")


def equal_weight_benchmark_return(
    previous_closes: Mapping[str, object],
    current_closes: Mapping[str, object],
) -> Decimal:
    returns: list[Decimal] = []

    for ticker in sorted(set(previous_closes) & set(current_closes)):
        previous = to_decimal(previous_closes[ticker])
        current = to_decimal(current_closes[ticker])

        if previous <= ZERO or current <= ZERO:
            continue

        returns.append(current / previous - ONE)

    if not returns:
        raise ValueError("No valid benchmark returns are available")

    return sum(returns, start=ZERO) / Decimal(len(returns))


def build_daily_performance_row(
    *,
    performance_date: object,
    broker: object,
    mark_prices: Mapping[str, object],
    previous_rows: pd.DataFrame,
    benchmark_return: Decimal | int | float | str,
    turnover: Decimal | int | float | str,
    skipped_trade_count: int,
) -> dict[str, object]:
    benchmark_daily_return = to_decimal(benchmark_return)
    turnover_value = to_decimal(turnover)

    market_value = sum(
        to_decimal(mark_prices.get(ticker, ZERO))
        * position.economic_quantity
        for ticker, position in broker.positions.items()
    )

    portfolio_value = (
        broker.settled_cash
        + broker.unsettled_cash
        + market_value
    )

    if portfolio_value < ZERO:
        raise ValueError("Portfolio value cannot be negative")

    if previous_rows.empty:
        daily_return = ZERO
        cumulative_return = ZERO
        benchmark_value = portfolio_value
        cumulative_benchmark_return = ZERO
        drawdown = ZERO
    else:
        previous = previous_rows.iloc[-1]
        previous_value = to_decimal(previous["portfolio_value"])
        previous_benchmark_value = to_decimal(
            previous["benchmark_value"]
        )

        if previous_value <= ZERO:
            raise ValueError(
                "Previous portfolio value must be positive"
            )

        if previous_benchmark_value <= ZERO:
            raise ValueError(
                "Previous benchmark value must be positive"
            )

        daily_return = portfolio_value / previous_value - ONE
        cumulative_return = (
            to_decimal(previous["cumulative_return"]) + ONE
        ) * (daily_return + ONE) - ONE

        benchmark_value = (
            previous_benchmark_value
            * (benchmark_daily_return + ONE)
        )
        cumulative_benchmark_return = (
            to_decimal(
                previous["cumulative_benchmark_return"]
            ) + ONE
        ) * (benchmark_daily_return + ONE) - ONE

        prior_values = [
            to_decimal(value)
            for value in previous_rows["portfolio_value"].tolist()
        ]
        running_peak = max(prior_values + [portfolio_value])
        drawdown = (
            portfolio_value / running_peak - ONE
            if running_peak > ZERO
            else ZERO
        )

    active_return = daily_return - benchmark_daily_return

    cash_value = broker.settled_cash + broker.unsettled_cash
    cash_weight = (
        cash_value / portfolio_value
        if portfolio_value > ZERO
        else ZERO
    )

    holdings_count = sum(
        1
        for position in broker.positions.values()
        if position.economic_quantity > 0
    )

    return {
        "date": str(performance_date),
        "portfolio_value": str(portfolio_value),
        "settled_cash": str(broker.settled_cash),
        "unsettled_cash": str(broker.unsettled_cash),
        "market_value": str(market_value),
        "daily_return": str(daily_return),
        "cumulative_return": str(cumulative_return),
        "benchmark_value": str(benchmark_value),
        "benchmark_return": str(benchmark_daily_return),
        "cumulative_benchmark_return": str(
            cumulative_benchmark_return
        ),
        "active_return": str(active_return),
        "drawdown": str(drawdown),
        "turnover": str(turnover_value),
        "cash_weight": str(cash_weight),
        "holdings_count": holdings_count,
        "skipped_trade_count": int(skipped_trade_count),
    }
