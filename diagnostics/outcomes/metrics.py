"""Pure, direction-aware underlying-price outcome calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .specs import Direction


@dataclass(frozen=True)
class PriceBar:
    market_date: date
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class OutcomeMetrics:
    directional_return: float
    mfe: float
    mae: float
    elapsed_calendar_days: int
    elapsed_trading_sessions: int


def directional_return(
    direction: Direction, entry_close: float, exit_close: float
) -> float:
    _positive_prices(entry_close, exit_close)
    if direction is Direction.LONG:
        return exit_close / entry_close - 1.0
    return entry_close / exit_close - 1.0


def excursions(
    direction: Direction, entry_close: float, bars: Iterable[PriceBar]
) -> tuple[float, float]:
    _positive_prices(entry_close)
    rows = tuple(bars)
    if not rows:
        raise ValueError("excursions require at least one forward price bar")
    _positive_prices(*(value for row in rows for value in (row.high, row.low)))
    if direction is Direction.LONG:
        return (
            max(row.high / entry_close - 1.0 for row in rows),
            min(row.low / entry_close - 1.0 for row in rows),
        )
    return (
        max(entry_close / row.low - 1.0 for row in rows),
        min(entry_close / row.high - 1.0 for row in rows),
    )


def calculate_metrics(
    direction: Direction,
    entry_date: date,
    entry_close: float,
    exit_bar: PriceBar,
    window: Iterable[PriceBar],
) -> OutcomeMetrics:
    rows = tuple(window)
    mfe, mae = excursions(direction, entry_close, rows)
    return OutcomeMetrics(
        directional_return(direction, entry_close, exit_bar.close),
        mfe,
        mae,
        (exit_bar.market_date - entry_date).days,
        len(rows),
    )


def _positive_prices(*prices: float) -> None:
    if any(price <= 0 for price in prices):
        raise ValueError("prices must be positive")
