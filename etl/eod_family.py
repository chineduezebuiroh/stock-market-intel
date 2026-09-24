"""Non-authoritative, run-scoped stock D/W/M observation families.

This module deliberately has no production publication or registry dependencies.
It acquires native observations, reconciles a daily Close from already acquired
evidence, and serializes compact shadow lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import pandas as pd

from core import storage
from core.paths import DATA
from etl.eod_quality import valid_stock_eod_rows
from etl.eod_reconciliation import (
    ReconciliationResult,
    canonical_terminal_session_close,
    observation_from_native_frame,
    repair_terminal_close,
)
from etl.sources import safe_load_eod


FAMILY_INTERVALS = ("1d", "1wk", "1mo")
TIMEFRAME_BY_INTERVAL = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}
DEFAULT_WINDOW_BARS = {"1d": 260, "1wk": 210, "1mo": 250}
PROVIDER_METADATA_KEYS = (
    "priceHint", "lastTrade", "dataGranularity", "range",
    "exchangeTimezoneName", "regularMarketTime", "regularMarketPrice",
    "currentTradingPeriod", "metadata_error",
)
MANIFEST_SCHEMA = (
    "record_type", "family_run_id", "symbol", "interval", "timeframe",
    "acquisition_status", "fetch_started_at", "fetch_completed_at",
    "terminal_session", "session_evidence", "provider", "provider_version",
    "adjustment_mode", "session_semantics", "provider_metadata_json",
    "payload_sha256", "quality_status", "quality_reason", "close_is_null",
    "raw_open", "raw_high", "raw_low", "raw_close", "raw_adj_close",
    "raw_volume", "reconciliation_eligible", "reconciliation_status",
    "reconciliation_tier", "canonical_source_intervals",
    "canonical_observed_close", "reconciliation_reason", "finalized_open",
    "finalized_high", "finalized_low", "finalized_close",
    "finalized_adj_close", "finalized_volume", "finalized_sha256",
    "repair_provenance_json", "acquisition_error", "requests_attempted",
    "requests_succeeded", "requests_failed", "requested_intervals",
    "reconciliation_provider_requests", "validity_pattern",
    "close_null_pattern", "d_to_w_gap_seconds", "w_to_m_gap_seconds",
    "family_duration_seconds",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value else None


def frame_sha256(frame: pd.DataFrame | None) -> str | None:
    """Hash canonical frame content deterministically (attributes excluded)."""
    if frame is None:
        return None
    payload = frame.to_json(date_format="iso", date_unit="ns", orient="split")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _terminal_values(frame: pd.DataFrame | None) -> dict[str, float | None]:
    names = ("open", "high", "low", "close", "adj_close", "volume")
    if frame is None or frame.empty:
        return {name: None for name in names}
    row = frame.iloc[-1]
    return {name: _number(row.get(name)) for name in names}


def _quality(frame: pd.DataFrame | None) -> tuple[str, str | None]:
    if frame is None or frame.empty:
        return "unavailable", "empty provider payload"
    mask = valid_stock_eod_rows(frame)
    if bool(mask.iloc[-1]):
        return "valid", None
    row = frame.iloc[-1]
    invalid = [name for name in ("open", "high", "low", "close", "volume")
               if _number(row.get(name)) is None]
    if _number(row.get("volume")) is not None and _number(row.get("volume")) < 0:
        invalid.append("volume_negative")
    return "invalid", "invalid terminal required fields: " + ",".join(invalid)


@dataclass(frozen=True)
class NativeEODObservation:
    source_interval: str
    raw_frame: pd.DataFrame | None
    fetch_started_at: datetime
    fetch_completed_at: datetime
    acquisition_status: str
    acquisition_error: str | None = None
    provider: str = "yahoo"
    provider_version: str | None = None
    adjustment_mode: str | None = None
    session_semantics: str | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    payload_sha256: str | None = None
    quality_status: str = "unavailable"
    quality_reason: str | None = None
    terminal_market_session: str | None = None
    session_evidence: str | None = None
    raw_terminal_values: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_interval not in FAMILY_INTERVALS:
            raise ValueError(f"unsupported family interval: {self.source_interval}")
        if self.raw_frame is not None:
            object.__setattr__(self, "raw_frame", self.raw_frame.copy(deep=True))
        object.__setattr__(self, "provider_metadata", dict(self.provider_metadata))
        object.__setattr__(self, "raw_terminal_values", dict(self.raw_terminal_values))


@dataclass(frozen=True)
class FinalizedEODObservation:
    native: NativeEODObservation
    finalized_frame: pd.DataFrame | None
    data_quality_status: str
    repair_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.finalized_frame is not None:
            object.__setattr__(self, "finalized_frame", self.finalized_frame.copy(deep=True))
        object.__setattr__(self, "repair_provenance", dict(self.repair_provenance))


@dataclass(frozen=True)
class RequestAccounting:
    attempted: int
    succeeded: int
    failed: int
    intervals: tuple[str, ...]
    reconciliation_provider_requests: int = 0


@dataclass(frozen=True)
class StockEODObservationFamily:
    run_id: str
    symbol: str
    daily: NativeEODObservation | None
    weekly: NativeEODObservation | None
    monthly: NativeEODObservation | None
    requests: RequestAccounting

    def __post_init__(self) -> None:
        expected = (("daily", "1d"), ("weekly", "1wk"), ("monthly", "1mo"))
        for member, interval in expected:
            observation = getattr(self, member)
            if observation is not None and observation.source_interval != interval:
                raise ValueError(f"{member} must contain {interval}")

    def observations(self) -> tuple[NativeEODObservation, ...]:
        return tuple(x for x in (self.daily, self.weekly, self.monthly) if x is not None)


@dataclass(frozen=True)
class FamilyFinalization:
    family: StockEODObservationFamily
    daily: FinalizedEODObservation | None
    weekly: FinalizedEODObservation | None
    monthly: FinalizedEODObservation | None
    reconciliation: ReconciliationResult | None


Loader = Callable[..., pd.DataFrame | None]


def _native_observation(
    interval: str, frame: pd.DataFrame | None, started: datetime, completed: datetime,
    error: str | None,
) -> NativeEODObservation:
    provenance = frame.attrs.get("eod_provenance", {}) if frame is not None else {}
    metadata = provenance.get("provider_metadata", {})
    metadata = ({key: metadata[key] for key in PROVIDER_METADATA_KEYS if key in metadata}
                if isinstance(metadata, dict) else {})
    quality_status, quality_reason = _quality(frame)
    terminal = observation_from_native_frame(frame) if frame is not None else None
    acquired = frame is not None and not frame.empty
    return NativeEODObservation(
        source_interval=interval, raw_frame=frame, fetch_started_at=started,
        fetch_completed_at=completed,
        acquisition_status="success" if acquired else "failed",
        acquisition_error=error or (None if acquired else "empty provider payload"),
        provider_version=provenance.get("yfinance_version"),
        adjustment_mode=provenance.get("adjustment_mode"),
        session_semantics=provenance.get("session_semantics"),
        provider_metadata=metadata,
        payload_sha256=provenance.get("payload_sha256") or frame_sha256(frame),
        quality_status=quality_status, quality_reason=quality_reason,
        terminal_market_session=(terminal.terminal_market_session_date.isoformat()
                                 if terminal and terminal.terminal_market_session_date else None),
        session_evidence=terminal.session_evidence if terminal else None,
        raw_terminal_values=_terminal_values(frame),
    )


def acquire_stock_eod_family(
    run_id: str, symbol: str, *, loader: Loader = safe_load_eod,
    session: str = "regular", window_bars: Mapping[str, int] | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> StockEODObservationFamily:
    """Sequentially acquire D/W/M exactly once each; isolate every failure."""
    windows = dict(DEFAULT_WINDOW_BARS if window_bars is None else window_bars)
    acquired: dict[str, NativeEODObservation] = {}
    succeeded = 0
    for interval in FAMILY_INTERVALS:
        started = clock()
        frame = None
        error = None
        try:
            frame = loader(symbol, timeframe=TIMEFRAME_BY_INTERVAL[interval],
                           window_bars=windows[interval], session=session)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        completed = clock()
        observation = _native_observation(interval, frame, started, completed, error)
        acquired[interval] = observation
        succeeded += observation.acquisition_status == "success"
    return StockEODObservationFamily(
        run_id=run_id, symbol=symbol.strip().upper(), daily=acquired["1d"],
        weekly=acquired["1wk"], monthly=acquired["1mo"],
        requests=RequestAccounting(3, succeeded, 3 - succeeded, FAMILY_INTERVALS),
    )


def reconcile_stock_eod_family(family: StockEODObservationFamily) -> FamilyFinalization:
    """Finalize already-acquired observations without invoking any provider."""
    native = {"1d": family.daily, "1wk": family.weekly, "1mo": family.monthly}
    finalized: dict[str, FinalizedEODObservation | None] = {}
    reconciliation = None
    for interval in ("1wk", "1mo"):
        item = native[interval]
        finalized[interval] = (FinalizedEODObservation(
            item, item.raw_frame, item.quality_status, {"status": "native_unmodified"}
        ) if item is not None else None)

    daily = family.daily
    if daily is None:
        finalized["1d"] = None
    elif daily.quality_status == "valid":
        finalized["1d"] = FinalizedEODObservation(
            daily, daily.raw_frame, "valid", {"status": "native_unmodified"}
        )
    elif daily.raw_frame is None or daily.raw_frame.empty:
        finalized["1d"] = FinalizedEODObservation(
            daily, daily.raw_frame, "excluded", {"reason": "daily unavailable"}
        )
    else:
        target_session = pd.Timestamp(daily.raw_frame.index[-1]).date()
        peers = [item.raw_frame for item in (family.weekly, family.monthly)
                 if item is not None and item.raw_frame is not None and not item.raw_frame.empty]
        observations = [observation_from_native_frame(frame) for frame in peers]
        tolerances = [obs.price_tolerance for obs in observations if obs.price_tolerance is not None]
        tolerance = max(tolerances, default=1e-4)
        reconciliation = canonical_terminal_session_close(
            observations, target_session=target_session, price_tolerance=tolerance
        )
        repaired, provenance = repair_terminal_close(daily.raw_frame, reconciliation)
        finalized["1d"] = FinalizedEODObservation(
            daily, repaired, provenance.get("data_quality_status", "excluded"), provenance
        )
    return FamilyFinalization(family, finalized["1d"], finalized["1wk"],
                              finalized["1mo"], reconciliation)


def validity_pattern(family: StockEODObservationFamily) -> tuple[str, str]:
    observations = (family.daily, family.weekly, family.monthly)
    states = ["valid" if item and item.quality_status == "valid" else "invalid"
              for item in observations]
    nulls = [bool(not item or item.raw_terminal_values.get("close") is None)
             for item in observations]
    return ("/".join(f"{i}:{s}" for i, s in zip(FAMILY_INTERVALS, states)),
            "/".join(f"{i}:close_null={str(n).lower()}" for i, n in zip(FAMILY_INTERVALS, nulls)))


def family_manifest(finalization: FamilyFinalization) -> pd.DataFrame:
    family = finalization.family
    pattern, null_pattern = validity_pattern(family)
    by_interval = {"1d": finalization.daily, "1wk": finalization.weekly,
                   "1mo": finalization.monthly}
    observations = {"1d": family.daily, "1wk": family.weekly, "1mo": family.monthly}
    d, w, m = family.daily, family.weekly, family.monthly
    d_w = (w.fetch_started_at - d.fetch_completed_at).total_seconds() if d and w else None
    w_m = (m.fetch_started_at - w.fetch_completed_at).total_seconds() if w and m else None
    duration = (m.fetch_completed_at - d.fetch_started_at).total_seconds() if d and m else None
    result = finalization.reconciliation
    rows = []
    for interval in FAMILY_INTERVALS:
        native = observations[interval]
        final = by_interval[interval]
        raw = native.raw_terminal_values if native else _terminal_values(None)
        final_values = _terminal_values(final.finalized_frame if final else None)
        row = {
            "record_type": "interval", "family_run_id": family.run_id,
            "symbol": family.symbol, "interval": interval,
            "timeframe": TIMEFRAME_BY_INTERVAL[interval],
            "acquisition_status": native.acquisition_status if native else "not_requested",
            "fetch_started_at": _iso(native.fetch_started_at) if native else None,
            "fetch_completed_at": _iso(native.fetch_completed_at) if native else None,
            "terminal_session": native.terminal_market_session if native else None,
            "session_evidence": native.session_evidence if native else None,
            "provider": native.provider if native else None,
            "provider_version": native.provider_version if native else None,
            "adjustment_mode": native.adjustment_mode if native else None,
            "session_semantics": native.session_semantics if native else None,
            "provider_metadata_json": _stable_json(native.provider_metadata) if native else "{}",
            "payload_sha256": native.payload_sha256 if native else None,
            "quality_status": native.quality_status if native else "unavailable",
            "quality_reason": native.quality_reason if native else "not requested",
            "close_is_null": raw["close"] is None,
            **{f"raw_{name}": raw[name] for name in raw},
            "reconciliation_eligible": bool(interval == "1d" and native and native.quality_status != "valid"),
            "reconciliation_status": result.status if interval == "1d" and result else "not_needed",
            "reconciliation_tier": result.evidence_tier if interval == "1d" and result else None,
            "canonical_source_intervals": "+".join(result.corroborating_sources) if interval == "1d" and result else None,
            "canonical_observed_close": result.canonical_close if interval == "1d" and result else None,
            "reconciliation_reason": result.reason if interval == "1d" and result else None,
            **{f"finalized_{name}": final_values[name] for name in final_values},
            "finalized_sha256": frame_sha256(final.finalized_frame) if final else None,
            "repair_provenance_json": _stable_json(final.repair_provenance) if final else "{}",
            "acquisition_error": native.acquisition_error if native else None,
            "requests_attempted": family.requests.attempted,
            "requests_succeeded": family.requests.succeeded,
            "requests_failed": family.requests.failed,
            "requested_intervals": ",".join(family.requests.intervals),
            "reconciliation_provider_requests": family.requests.reconciliation_provider_requests,
            "validity_pattern": pattern, "close_null_pattern": null_pattern,
            "d_to_w_gap_seconds": d_w, "w_to_m_gap_seconds": w_m,
            "family_duration_seconds": duration,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=MANIFEST_SCHEMA)


def shadow_manifest_path(run_id: str, *, data_root: Path = DATA) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and dash")
    return data_root / "_shadow" / "eod_family" / run_id / "manifest.parquet"


def write_shadow_manifest(frame: pd.DataFrame, run_id: str, *, data_root: Path = DATA,
                          writer: Callable[[pd.DataFrame, Path], None] | None = None) -> Path:
    path = shadow_manifest_path(run_id, data_root=data_root)
    if writer is None:
        writer = storage.save_parquet
    writer(frame.loc[:, MANIFEST_SCHEMA], path)
    return path
