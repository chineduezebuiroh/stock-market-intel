"""Fail-closed reconciliation of independent native stock EOD observations.

Only native Yahoo 1d/1wk/1mo observations are eligible.  The caller must
establish the actual terminal market session; bucket labels alone are not
session evidence for weekly/monthly bars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
from typing import Any, Iterable

import pandas as pd

ELIGIBLE_INTERVALS = frozenset({"1d", "1wk", "1mo"})


@dataclass(frozen=True)
class TerminalObservation:
    source_interval: str
    interval_label: str
    terminal_market_session_date: date | None
    session_evidence: str | None
    close: float | None
    adjustment_mode: str = "raw_unadjusted"
    session_semantics: str = "regular"
    independent_native: bool = True
    revision_ambiguous: bool = False
    price_tolerance: float | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    status: str
    canonical_close: float | None
    corroborating_sources: tuple[str, ...]
    conflicting_sources: tuple[str, ...]
    price_tolerance: float
    reason: str
    evidence_tier: str | None = None

    def provenance(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _metadata_price_tolerance(metadata: Any) -> float:
    try:
        hint = int(metadata.get("priceHint", 4))
        if isinstance(metadata.get("priceHint"), bool) or not 0 <= hint <= 12:
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        hint = 4
    return max(1e-12, 0.5 * 10 ** (-hint))


def observation_from_native_frame(frame: pd.DataFrame) -> TerminalObservation:
    """Translate a provenance-bearing native frame without inventing sessions."""
    provenance = frame.attrs.get("eod_provenance", {}) if frame is not None else {}
    interval = str(provenance.get("source_interval", ""))
    if frame is None or frame.empty:
        return TerminalObservation(interval, "", None, None, None)
    label = pd.Timestamp(frame.index[-1])
    close = _finite(frame.iloc[-1].get("close"))
    evidence = None
    session_date = None
    metadata = provenance.get("provider_metadata", {})
    if interval == "1d":
        session_date = label.date()
        evidence = "native_daily_interval_label"
    elif interval in {"1wk", "1mo"}:
        # yfinance adds lastTrade only when its interval-specific cleanup merges
        # a separate live row into this aggregate.  Price equality binds that
        # evidence to the returned terminal aggregate.
        last_trade = metadata.get("lastTrade") if isinstance(metadata, dict) else None
        if isinstance(last_trade, dict):
            last_price = _finite(last_trade.get("Price"))
            try:
                last_time = pd.Timestamp(last_trade.get("Time"))
            except Exception:
                last_time = pd.NaT
            if (close is not None and last_price is not None
                    and abs(close - last_price) <= _metadata_price_tolerance(metadata)
                    and not pd.isna(last_time)):
                session_date = last_time.date()
                evidence = "yfinance_interval_merge_lastTrade"
    return TerminalObservation(
        source_interval=interval,
        interval_label=label.isoformat(),
        terminal_market_session_date=session_date,
        session_evidence=evidence,
        close=close,
        adjustment_mode=str(provenance.get("adjustment_mode", "unknown")),
        session_semantics=str(provenance.get("session_semantics", "unknown")),
        price_tolerance=_metadata_price_tolerance(metadata) if isinstance(metadata, dict) else None,
    )


def canonical_terminal_session_close(
    observations: Iterable[TerminalObservation],
    *,
    target_session: date,
    price_tolerance: float,
) -> ReconciliationResult:
    """Return a canonical raw close under tier-1 consensus or strict tier 2.

    Every eligible source must be independent, raw/unadjusted, regular-session,
    explicitly complete through ``target_session``, and free of known revision
    ambiguity. Any eligible same-session contradiction vetoes repair. Tier 2 is
    limited to one native W/M source with price-bound interval merge evidence.
    """
    if not math.isfinite(price_tolerance) or price_tolerance < 0:
        raise ValueError("price_tolerance must be finite and non-negative")
    eligible: list[tuple[TerminalObservation, float]] = []
    for observation in observations:
        value = _finite(observation.close)
        if (
            observation.source_interval in ELIGIBLE_INTERVALS
            and observation.independent_native
            and observation.terminal_market_session_date == target_session
            and observation.session_evidence
            and observation.adjustment_mode == "raw_unadjusted"
            and observation.session_semantics == "regular"
            and not observation.revision_ambiguous
            and value is not None
        ):
            eligible.append((observation, value))
    if len(eligible) < 2:
        if len(eligible) == 1:
            observation, value = eligible[0]
            if (
                observation.source_interval in {"1wk", "1mo"}
                and observation.session_evidence == "yfinance_interval_merge_lastTrade"
            ):
                return ReconciliationResult(
                    "canonical", value, (observation.source_interval,), (),
                    price_tolerance,
                    "single native close with price-bound interval-specific lastTrade",
                    "tier_2_provenance_bound_single_source",
                )
        return ReconciliationResult("excluded", None, (), (), price_tolerance,
                                    "no eligible native close", None)

    # A consensus cluster must include at least two values.  Any other eligible
    # value outside that cluster is an unresolved contradiction and vetoes it.
    for anchor_observation, anchor in eligible:
        agreeing = [(o, v) for o, v in eligible if abs(v - anchor) <= price_tolerance]
        if len(agreeing) >= 2:
            conflicts = tuple(sorted(o.source_interval for o, v in eligible
                                     if abs(v - anchor) > price_tolerance))
            if conflicts:
                return ReconciliationResult(
                    "conflict", None,
                    tuple(sorted(o.source_interval for o, _ in agreeing)), conflicts,
                    price_tolerance, "eligible same-session close disagreement", None,
                )
            sources = tuple(sorted(o.source_interval for o, _ in agreeing))
            # Prefer the observation with the finest declared price precision.
            # Equal-precision observations use the finer native interval because
            # it has the smallest aggregation/revision surface. Always return an
            # observed value; never synthesize a midpoint.
            selected_observation, value = min(
                agreeing,
                key=lambda item: (
                    item[0].price_tolerance
                    if item[0].price_tolerance is not None
                    else price_tolerance,
                    ("1d", "1wk", "1mo").index(item[0].source_interval),
                ),
            )
            return ReconciliationResult("canonical", value, sources, (), price_tolerance,
                                        "independent native same-session consensus; selected "
                                        f"{selected_observation.source_interval} by precision/finer-interval precedence",
                                        "tier_1_independent_consensus")
    return ReconciliationResult(
        "conflict", None, (), tuple(sorted(o.source_interval for o, _ in eligible)),
        price_tolerance, "no two eligible closes agree within tolerance", None,
    )


def repair_terminal_close(
    frame: pd.DataFrame,
    reconciliation: ReconciliationResult,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repair only terminal raw Close; never synthesize Adj Close or OHLCV."""
    provenance = reconciliation.provenance()
    if frame is None or frame.empty or reconciliation.status != "canonical":
        provenance.update(data_quality_status="excluded", close_status="malformed")
        return frame, provenance
    row = frame.iloc[-1]
    values = {name: _finite(row.get(name)) for name in ("open", "high", "low", "volume")}
    close = _finite(reconciliation.canonical_close)
    if any(value is None for value in values.values()) or values["volume"] < 0:
        provenance.update(data_quality_status="excluded", close_status="malformed",
                          reason="target O/H/L/V is malformed")
        return frame, provenance
    if (
        values["high"] < values["low"]
        or not values["low"] <= values["open"] <= values["high"]
        or close is None
        or not values["low"] <= close <= values["high"]
    ):
        provenance.update(data_quality_status="excluded", close_status="malformed",
                          reason="canonical close violates target high/low")
        return frame, provenance
    repaired = frame.copy()
    repaired.iloc[-1, repaired.columns.get_loc("close")] = close
    provenance.update(
        data_quality_status="repaired", close_status="repaired",
        close_source="+".join(reconciliation.corroborating_sources),
        close_repair_method="native_terminal_session_consensus",
        raw_observed_close=_finite(row.get("close")), repaired_close=close,
    )
    return repaired, provenance
