"""Typed combo horizons, policy modes, and canonical artifact schema."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Mapping


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PolicyMode(str, Enum):
    HISTORICAL_PRODUCTION = "HISTORICAL_PRODUCTION"
    COUNTERFACTUAL_PARTICIPATION = "COUNTERFACTUAL_PARTICIPATION"


class MaturityStatus(str, Enum):
    MATURE = "MATURE"
    IMMATURE = "IMMATURE"
    MISSING_PRICE_HISTORY = "MISSING_PRICE_HISTORY"
    UNRESOLVABLE_TARGET_DATE = "UNRESOLVABLE_TARGET_DATE"


class CoverageStatus(str, Enum):
    MATURE = "MATURE"
    IMMATURE = "IMMATURE"
    MISSING_SYMBOL_HISTORY = "MISSING_SYMBOL_HISTORY"
    ENTRY_PREDATES_RETAINED_HISTORY = "ENTRY_PREDATES_RETAINED_HISTORY"
    UNRESOLVABLE_TARGET_DATE = "UNRESOLVABLE_TARGET_DATE"
    INCOMPLETE_EXCURSION_WINDOW = "INCOMPLETE_EXCURSION_WINDOW"
    INVALID_PRICE_DATA = "INVALID_PRICE_DATA"


@dataclass(frozen=True)
class Horizon:
    horizon_id: str
    multiple: int
    calendar_days: int
    primary_directions: frozenset[Direction]


@dataclass(frozen=True)
class ComboHorizons:
    combo: str
    upper_timeframe: str
    horizons: tuple[Horizon, ...]
    alignment_tolerance_days: int = 7


def _horizons(prefix: str, days: tuple[int, int, int]) -> tuple[Horizon, ...]:
    return tuple(
        Horizon(
            f"{prefix}_{multiple}x",
            multiple,
            day_count,
            frozenset({Direction.SHORT} if multiple == 1 else {Direction.LONG}),
        )
        for multiple, day_count in enumerate(days, start=1)
    )


HORIZON_SPECS = {
    "stocks_c_dwm_all": ComboHorizons(
        "stocks_c_dwm_all", "MONTH", _horizons("DWM", (30, 60, 90))
    ),
    "stocks_b_wmq_all": ComboHorizons(
        "stocks_b_wmq_all", "QUARTER", _horizons("WMQ", (90, 180, 270))
    ),
    "stocks_a_mqy_all": ComboHorizons(
        "stocks_a_mqy_all", "YEAR", _horizons("MQY", (365, 730, 1095))
    ),
}


@dataclass(frozen=True)
class EntryState:
    """Immutable values copied from one canonical combo-history observation."""

    combo: str
    direction: Direction
    symbol: str
    entry_market_date: date
    entry_execution_timestamp: datetime
    source_combo_history_key: str
    engine_regime_id: str
    preparticipation_qualified: bool
    historical_participation_pass: bool
    historical_participation_route: str
    lower_sigvol: float | None
    middle_sigvol: float | None
    lower_ratio: float | None
    middle_ratio: float | None
    upper_available: bool
    historical_long_score: float
    historical_short_score: float
    entry_close: float


@dataclass(frozen=True)
class PolicyEvaluation:
    mode: PolicyMode
    entry_state: EntryState
    participation_pass: bool
    participation_route: str
    parameters: Mapping[str, float]


def evaluate_policy(
    entry: EntryState,
    mode: PolicyMode,
    *,
    participation_pass: bool | None = None,
    participation_route: str | None = None,
    parameters: Mapping[str, float] | None = None,
) -> PolicyEvaluation:
    """Apply policy without mutating or recomputing immutable entry state."""
    if mode is PolicyMode.HISTORICAL_PRODUCTION:
        if (
            participation_pass is not None
            or participation_route is not None
            or parameters
        ):
            raise ValueError("historical policy cannot accept counterfactual overrides")
        return PolicyEvaluation(
            mode,
            entry,
            entry.historical_participation_pass,
            entry.historical_participation_route,
            {},
        )
    if participation_pass is None:
        raise ValueError(
            "counterfactual participation requires an evaluated pass value"
        )
    return PolicyEvaluation(
        mode,
        replace(entry),
        participation_pass,
        participation_route or "NONE",
        dict(parameters or {}),
    )


OUTCOME_COLUMNS = (
    *EntryState.__dataclass_fields__,
    "policy_mode",
    "evaluated_participation_pass",
    "evaluated_participation_route",
    "horizon_id",
    "horizon_multiple",
    "target_calendar_date",
    "resolved_exit_date",
    "maturity_status",
    "exit_close",
    "directional_return",
    "mfe",
    "mae",
    "elapsed_calendar_days",
    "elapsed_trading_sessions",
    "outcome_price_source_key",
    "outcome_price_asof",
)


def target_date(entry_date: date, horizon: Horizon) -> date:
    return entry_date + timedelta(days=horizon.calendar_days)
