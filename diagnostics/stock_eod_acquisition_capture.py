"""OFF-by-default, bounded local evidence; never an input to production policy.

B/C snapshots contain detached scalar representations only. JSON encoding and
local writes occur after loader completion, outside its SIGALRM. Raw interception
is deliberately unavailable: get/cache_get does not expose internal retries and
response lineage sufficiently to satisfy the approved attribution contract.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import time
from zoneinfo import available_timezones

import numpy as np
import pandas as pd

from core.paths import DATA

_ACTIVE = ContextVar("stock_eod_diagnostic_collector", default=None)
_FETCH = ContextVar("stock_eod_diagnostic_fetch", default=None)
QUOTE_NAMES = {"open", "high", "low", "close", "adj close", "adj_close", "volume"}
META_KEYS = {"priceHint", "dataGranularity", "exchangeTimezoneName", "regularMarketTime",
             "regularMarketPrice", "currentTradingPeriod", "lastTrade", "range", "metadata_error"}
_TIMEZONES = frozenset({"America/New_York", "UTC"})
PROVENANCE_KEYS = {"source_interval", "provider_fetch_time", "session_semantics", "adjustment_mode", "yfinance_version"}
FINGERPRINTS = {
    "get": "cca1000b69b0438cabc6e96b302faa5e55211387dca045a455acd7d23eb526e2",
    "cache_get": "787626d085122ee04abbf80ed332a282043c1778dedc2e124f2c9a0a9a8a6699",
}


def _timeout(exc):
    return isinstance(exc, TimeoutError) or (
        type(exc).__name__ == "TimeoutException" and type(exc).__module__ == "etl.sources"
    )


def attempt(fn, *args, **kwargs):
    """Capture errors are isolated; acquisition interrupts are never swallowed."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if _timeout(exc):
            raise
        return None


def atom(value):
    """Do not call repr/str on arbitrary objects or emit unexpected string values."""
    if value is None:
        return {"kind": "None"}
    if value is pd.NA:
        return {"kind": "pd.NA"}
    if value is pd.NaT:
        return {"kind": "NaT"}
    if isinstance(value, (pd.Timestamp, datetime)):
        return {"kind": "timestamp", "value": value.isoformat()}
    if isinstance(value, (bool, np.bool_)):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        # Guard pathological object cells carrying arbitrarily large integers.
        number = int(value)
        return {"kind": "integer", "value": number} if number.bit_length() <= 128 else {"kind": "oversized_integer"}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return {"kind": "NaN"}
        if math.isinf(number):
            return {"kind": "+infinity" if number > 0 else "-infinity"}
        return {"kind": "finite", "value": number, "hex": number.hex()}
    return {"kind": "unsupported_or_redacted"}


def label(value, depth=0):
    if depth > 4:
        return {"kind": "unsupported_or_redacted"}
    if isinstance(value, str):
        return value[:128]
    if isinstance(value, tuple):
        return [label(v, depth + 1) for v in value[:4]]
    return atom(value)


def metadata(mapping):
    """Whitelist both names and value types; never traverse unrestricted attrs."""
    if type(mapping) is not dict:
        return {}
    result = {}
    for key in sorted(META_KEYS):
        if key not in mapping:
            continue
        value = mapping[key]
        if key in {"exchangeTimezoneName", "dataGranularity", "range"}:
            accepted = _TIMEZONES if key == "exchangeTimezoneName" else {"1d", "1wk", "1mo"} if key == "dataGranularity" else {"", "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
            if isinstance(value, str) and value in accepted:
                result[key] = value
        elif key == "metadata_error":
            result[key] = {"kind": "present_redacted"}
        elif key == "regularMarketTime":
            result[key] = date_value(value)
        elif key == "lastTrade" and type(value) is dict:
            result[key] = {"Price": atom(value.get("Price")), "Time": date_value(value.get("Time"))}
        elif key == "currentTradingPeriod" and type(value) is dict:
            result[key] = {name: {edge: date_value(value[name].get(edge)) for edge in ("start", "end")}
                           for name in ("pre", "regular", "post") if type(value.get(name)) is dict}
        else:
            result[key] = atom(value)
    return result


def date_value(value):
    if isinstance(value, str):
        # Restrict retained strings to parseable timestamps.
        if len(value) > 64:
            return {"kind": "unsupported_or_redacted"}
        try:
            return atom(pd.Timestamp(value))
        except (ValueError, TypeError):
            return {"kind": "unsupported_or_redacted"}
    return atom(value)


def snapshot(frame):
    if frame is None:
        return {"state": "None"}
    if not isinstance(frame, pd.DataFrame):
        return {"state": "unsupported"}
    if len(frame.columns) > 32:
        return {"state": "gap", "reason": "too_many_columns", "rows": len(frame), "columns": len(frame.columns)}
    columns = list(frame.columns)
    index_name = frame.index.name
    result = {
        "state": "captured", "rows": len(frame), "columns": [label(c) for c in columns],
        "dtypes": [str(t) for t in frame.dtypes], "index_type": type(frame.index).__name__,
        "index_name": label(index_name), "index_timezone": str(getattr(frame.index, "tz", None)),
        "index_dtype": str(frame.index.dtype), "terminal_rows": [],
    }
    indices = [i for i, c in enumerate(columns) if (
        c[0] if isinstance(c, tuple) and c else c
    ).lower() in QUOTE_NAMES] if all(isinstance(c, str) or (isinstance(c, tuple) and c and isinstance(c[0], str)) for c in columns) else []
    for row in range(max(0, len(frame) - 2), len(frame)):
        result["terminal_rows"].append({"label": label(frame.index[row]),
            "values": {str(i): atom(frame.iat[row, i]) for i in indices}})
    present = {str(c[0] if isinstance(c, tuple) and c else c).lower() for c in columns}
    result["absent_quote_columns"] = sorted(QUOTE_NAMES - present)
    # No dataframe slice/copy: pandas may deepcopy unrestricted attrs on slicing.
    attrs = frame.attrs
    provenance = attrs.get("eod_provenance") if type(attrs) is dict else None
    if type(provenance) is dict:
        expected = {"source_interval": {"1d", "1wk", "1mo"}, "session_semantics": {"regular"},
                    "adjustment_mode": {"raw_unadjusted"}, "yfinance_version": {"1.7.0"}}
        result["provenance"] = {}
        for key in sorted(PROVENANCE_KEYS):
            if key not in provenance:
                continue
            value = provenance[key]
            if key == "provider_fetch_time":
                result["provenance"][key] = date_value(value)
            else:
                result["provenance"][key] = value if isinstance(value, str) and value in expected[key] else {"kind": "unsupported_or_redacted"}
        result["provider_metadata"] = metadata(provenance.get("provider_metadata"))
        digest = provenance.get("payload_sha256")
        if isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
            result["normalized_frame_sha256"] = digest
    return result


@dataclass(frozen=True)
class Config:
    application: bool = False
    raw_response: bool = False
    output_dir: str | None = None
    selected_symbols: tuple[str, ...] = ()
    per_fetch_bytes: int = 64 * 1024
    memory_bytes: int = 16 * 1024 * 1024
    run_bytes: int = 32 * 1024 * 1024
    slots: dict = field(default_factory=lambda: {"selected": 96, "boundary": 192, "malformed": 192, "controls": 96, "background": 24})
    boundary_seconds: int = 120
    background_modulus: int = 100

    def validate(self):
        if not (256 <= self.per_fetch_bytes <= 64 * 1024 and self.per_fetch_bytes <= self.memory_bytes <= 16 * 1024 * 1024):
            raise ValueError("invalid capture memory limits")
        if not 256 <= self.run_bytes <= 32 * 1024 * 1024:
            raise ValueError("invalid run byte limit")
        if set(self.slots) != {"selected", "boundary", "malformed", "controls", "background"} or any(type(v) is not int or not 0 <= v <= 512 for v in self.slots.values()):
            raise ValueError("invalid category quotas")
        if len(self.selected_symbols) > 32 or not 0 <= self.boundary_seconds <= 300 or self.background_modulus < 1:
            raise ValueError("invalid sampling parameters")


def raw_capability():
    """No hook is installed unless every approved raw safety requirement is met."""
    import yfinance
    from yfinance.data import YfData
    if yfinance.__version__ != "1.7.0":
        return "unsupported_yfinance_version"
    for name, expected in FINGERPRINTS.items():
        source = inspect.getsource(inspect.unwrap(getattr(YfData, name)))
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            return "unsupported_yfinance_source"
    return "get_cache_get_seam_does_not_expose_internal_retry_lineage"


class Collector:
    def __init__(self, config):
        global _TIMEZONES
        config.validate()
        # Resolve identifiers outside acquisition, not from the timed snapshot.
        _TIMEZONES = frozenset(available_timezones())
        self.config = config
        self.sequence = 0
        self.counts = Counter()
        self.gaps = Counter()
        self.pending = []
        self.pending_bytes = 0
        self.records = {category: [] for category in config.slots}
        self.bytes = 0
        self.last_malformed = False
        self.previous_success = []
        self.raw_status = "OFF"
        if config.raw_response:
            self.raw_status = attempt(raw_capability) or "raw_safety_check_failed"
        self.capture_ns = 0
        self.capture_calls = 0

    def begin(self, run_id, symbol, interval):
        self.sequence += 1
        return {"identity": {"family_run_id": run_id, "symbol": symbol, "native_interval": interval,
                             "acquisition_sequence": self.sequence},
                "started_at": datetime.now(timezone.utc).isoformat(), "boundaries": {}}

    def capture(self, stage, frame, facts):
        fetch = _FETCH.get()
        if fetch is None or not self.config.application:
            return
        started = time.perf_counter_ns()
        try:
            # A snapshot has at most 32 columns * 2 rows and constrained attrs.
            fetch["boundaries"][stage] = snapshot(frame)
            fetch["boundaries"][stage]["captured_at"] = datetime.now(timezone.utc).isoformat()
            allowed_facts = {}
            for key in ("requested_start", "requested_end"):
                if key in facts:
                    allowed_facts[key] = date_value(facts[key])
            for key in ("window_bars", "duplicates_removed", "auto_adjust", "actions"):
                if key in facts:
                    allowed_facts[key] = atom(facts[key])
            if facts.get("native_interval") in {"1d", "1wk", "1mo"}:
                allowed_facts["native_interval"] = facts["native_interval"]
            if "inserted_columns" in facts:
                allowed_facts["inserted_columns"] = [c for c in facts["inserted_columns"][:6] if c in QUOTE_NAMES]
            fetch.setdefault("stage_facts", {})[stage] = allowed_facts
            return True
        finally:
            self.capture_ns += time.perf_counter_ns() - started
            self.capture_calls += 1

    def complete(self, fetch):
        # Called after safe_load_eod's SIGALRM has been cancelled.
        fetch["completed_at"] = datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(fetch, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > self.config.per_fetch_bytes:
            self.gaps["oversized_fetch"] += 1
            return
        if self.pending_bytes + len(encoded) > self.config.memory_bytes // 4:
            self.gaps["pending_memory_limit"] += 1
            return
        self.pending.append(fetch)
        self.pending_bytes += len(encoded)
        return True

    def drain_symbol(self, family):
        # All intervals are now acquired; no callback can become peer evidence.
        by_interval = {"1d": family.daily, "1wk": family.weekly, "1mo": family.monthly}
        malformed = any(x is not None and x.quality_status != "valid" for x in by_interval.values())
        selected = family.symbol in self.config.selected_symbols
        boundary = any(near_boundary(item, self.config.boundary_seconds) for item in self.pending)
        background = int(hashlib.sha256((family.run_id + family.symbol).encode()).hexdigest()[:8], 16) % self.config.background_modulus == 0
        category = "selected" if selected else "boundary" if boundary else "malformed" if malformed else "controls" if self.last_malformed else "background" if background else None
        for item in self.pending:
            observation = by_interval[item["identity"]["native_interval"]]
            item["native_quality_status"] = observation.quality_status if observation is not None else "unavailable"
            item["acquisition_status"] = observation.acquisition_status if observation is not None else "not_requested"
        candidates = []
        for item in self.pending:
            encoded = json.dumps(item, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
            if len(encoded) > self.config.per_fetch_bytes:
                self.gaps["oversized_promoted_fetch"] += 1
            else:
                candidates.append((item, encoded))
        if len(candidates) != 3:
            self.gaps["incomplete_family_records"] += 1
        self.pending.clear()
        self.pending_bytes = 0
        if malformed and not self.last_malformed and self.previous_success:
            self.admit("controls", self.previous_success)
        self.last_malformed = malformed
        self.previous_success = [] if malformed else [encoded for _, encoded in candidates]
        if category is None:
            return True
        # Admit complete acquired families together, never just D at quota edge.
        self.admit(category, [encoded for _, encoded in candidates])
        return True

    def admit(self, category, candidates):
        size = sum(len(encoded) + 1 for encoded in candidates)
        # Leave space for detached snapshots and transient encodings. Encoded
        # retained bytes are capped at one quarter of the configured memory cap.
        reserved_limit = min(max(0, self.config.run_bytes - 64 * 1024), self.config.memory_bytes // 4) // len(self.records)
        if self.counts[category] + len(candidates) > self.config.slots[category]:
            self.gaps[category + "_quota"] += 1
        elif size + sum(len(x) + 1 for x in self.records[category]) > reserved_limit:
            self.gaps[category + "_byte_quota"] += 1
        else:
            self.records[category].extend(candidates)
            self.counts[category] += len(candidates)
            self.bytes += size

    def finish(self):
        if self.config.output_dir is None:
            return
        root = Path(self.config.output_dir).resolve()
        data_root = DATA.resolve()
        if root == data_root or data_root in root.parents or str(self.config.output_dir).startswith("s3:"):
            raise ValueError("diagnostic output must be outside production data")
        # A fresh private directory prevents overwriting prior evidence or following
        # a pre-existing evidence-file symlink. No production storage APIs are used.
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = Path(tempfile.mkdtemp(prefix="acquisition-", dir=root))
        for category, records in self.records.items():
            with (destination / (category + ".jsonl")).open("xb") as stream:
                os.chmod(stream.name, 0o600)
                for record in records:
                    stream.write(record + b"\n")
        summary = {"raw_response_status": self.raw_status, "application": self.config.application,
                   "counts": dict(self.counts), "capture_gaps": dict(self.gaps), "evidence_bytes": self.bytes,
                   "capture_calls": self.capture_calls, "capture_nanoseconds": self.capture_ns,
                   "limits": {"per_fetch_bytes": self.config.per_fetch_bytes, "memory_bytes": self.config.memory_bytes,
                              "run_bytes": self.config.run_bytes, "slots": self.config.slots}}
        (destination / "summary.json").write_text(json.dumps(summary, sort_keys=True))
        os.chmod(destination / "summary.json", 0o600)


def near_boundary(fetch, seconds):
    timestamp = pd.Timestamp(fetch["started_at"])
    distance = min((timestamp - timestamp.normalize()).total_seconds(),
                   (timestamp.normalize() + pd.Timedelta(days=1) - timestamp).total_seconds())
    if distance <= seconds:
        return True
    meta = fetch["boundaries"].get("C", {}).get("provider_metadata", {})
    periods = meta.get("currentTradingPeriod", {})
    for period in periods.values():
        for edge in period.values():
            if edge.get("kind") == "timestamp":
                boundary = pd.Timestamp(edge["value"])
                if boundary.tzinfo is not None and abs((timestamp - boundary).total_seconds()) <= seconds:
                    return True
    return False


@contextmanager
def collecting(config):
    collector = attempt(Collector, config) if config.application or config.raw_response else None
    token = _ACTIVE.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE.reset(token)
        if collector is not None:
            attempt(collector.finish)


@contextmanager
def acquisition(run_id, symbol, interval):
    collector = _ACTIVE.get()
    fetch = attempt(collector.begin, run_id, symbol, interval) if collector is not None else None
    token = _FETCH.set(fetch)
    try:
        yield
    finally:
        _FETCH.reset(token)
        if fetch is not None:
            if attempt(collector.complete, fetch) is None:
                collector.gaps["complete_error_or_gap"] += 1


def capture(stage, frame, **facts):
    collector = _ACTIVE.get()
    if collector is not None:
        if attempt(collector.capture, stage, frame, facts) is None and collector.config.application and _FETCH.get() is not None:
            # Count is bounded by the number of acquisitions, not error strings.
            collector.gaps["capture_error_or_noop"] += 1


def acquired_family(family):
    collector = _ACTIVE.get()
    if collector is not None:
        if attempt(collector.drain_symbol, family) is None:
            collector.pending.clear()
            collector.pending_bytes = 0
            collector.previous_success = []
            collector.last_malformed = False
            collector.gaps["family_promotion_error"] += 1


def configured_family(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Direct fixture/API callers can supply their own collection context.
        if _ACTIVE.get() is not None:
            return fn(*args, **kwargs)
        config = Config(application=os.getenv("STOCK_EOD_CAPTURE_APPLICATION") == "1",
                        raw_response=os.getenv("STOCK_EOD_CAPTURE_RAW") == "1",
                        output_dir=os.getenv("STOCK_EOD_CAPTURE_DIR"),
                        selected_symbols=tuple(os.getenv("STOCK_EOD_CAPTURE_SYMBOLS", "").upper().split(",")) if os.getenv("STOCK_EOD_CAPTURE_SYMBOLS") else ())
        with collecting(config):
            return fn(*args, **kwargs)
    return wrapped
