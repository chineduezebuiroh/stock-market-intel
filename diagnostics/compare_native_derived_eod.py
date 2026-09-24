"""Capture read-only evidence about Yahoo native D/W/M terminal stock bars.

This module deliberately has no dependency on project storage or orchestration.  Its
only side effect is writing files below the explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

INTERVALS = ("1d", "1wk", "1mo")
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
DEFAULT_SYMBOLS = ("AAPL", "MSFT", "SPY", "CRWD", "NVDA", "JPM", "XOM")
ET = ZoneInfo("America/New_York")


def iso_utc(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert provider metadata (including timestamps/numpy values) to JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns={name: name.lower().replace(" ", "_") for name in FIELDS})
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("America/New_York").tz_localize(None)
    columns = [name.lower().replace(" ", "_") for name in FIELDS]
    for column in columns:
        if column not in out:
            out[column] = np.nan
    return out[columns].sort_index()


def derive_terminal(daily: pd.DataFrame, interval: str) -> pd.Series:
    """Secondary evidence only: aggregate daily rows in the current W/M bucket."""
    if daily.empty:
        raise ValueError("daily payload is empty")
    terminal = daily.index.max()
    if interval == "1wk":
        start = terminal - pd.Timedelta(days=terminal.weekday())
    elif interval == "1mo":
        start = terminal.replace(day=1)
    else:
        raise ValueError(interval)
    current = daily.loc[(daily.index.normalize() >= start.normalize()) & (daily.index <= terminal)]
    return pd.Series(
        {
            "open": current.open.iloc[0],
            "high": current.high.max(),
            "low": current.low.min(),
            "close": current.close.iloc[-1],
            "adj_close": current.adj_close.iloc[-1],
            "volume": current.volume.sum(),
        }
    )


def payload_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_json(date_format="iso", orient="split").encode()).hexdigest()


def price_tolerance(metadata: dict[str, Any]) -> float:
    """Half of the provider's displayed price unit, with a tiny float floor."""
    try:
        hint = int(metadata.get("priceHint", 4))
    except (TypeError, ValueError):
        hint = 4
    hint = min(max(hint, 0), 12)
    return max(1e-12, 0.5 * 10 ** (-hint))


def close_differences(closes: dict[str, float | None]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for left, right in (("1d", "1wk"), ("1d", "1mo"), ("1wk", "1mo")):
        a, b = closes.get(left), closes.get(right)
        key = f"{left}_vs_{right}"
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            result[key] = {"absolute": None, "relative": None}
            continue
        absolute = abs(a - b)
        result[key] = {
            "absolute": absolute,
            "relative": absolute / max(abs(a), abs(b)) if max(abs(a), abs(b)) else 0.0,
        }
    return result


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def corroborates_completion(native: dict[str, Any], derived: dict[str, Any], tolerance: float) -> bool:
    """Use O/H/L/V, deliberately excluding Close and Adj Close, as corroboration."""
    for field in ("open", "high", "low"):
        a, b = _finite_number(native.get(field)), _finite_number(derived.get(field))
        if a is None or b is None or abs(a - b) > tolerance:
            return False
    a, b = _finite_number(native.get("volume")), _finite_number(derived.get("volume"))
    return a is not None and b is not None and abs(a - b) <= 0.5


def classify_observation(intervals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if any(intervals.get(i, {}).get("status") != "ok" for i in INTERVALS):
        return {"classification": "malformed/missing interval", "comparable": False}
    closes = {i: _finite_number(intervals[i]["terminal"].get("close")) for i in INTERVALS}
    if any(value is None for value in closes.values()):
        return {"classification": "malformed/missing interval", "comparable": False}
    # Native W/M labels identify bucket starts, not their last incorporated session.
    # Require independent non-Close aggregate corroboration before comparing closes.
    if not all(intervals[i].get("completion_corroboration") is True for i in ("1wk", "1mo")):
        return {"classification": "cannot establish comparable terminal session", "comparable": False}
    tolerance = max(float(intervals[i]["price_tolerance"]) for i in INTERVALS)
    values = list(closes.values())
    spread = max(values) - min(values)  # type: ignore[arg-type]
    if spread == 0:
        classification = "D/W/M exact agreement"
    elif spread <= tolerance:
        classification = "agreement within deterministic tolerance"
    else:
        classification = "disagreement"
    return {"classification": classification, "comparable": True, "close_spread": spread, "tolerance": tolerance}


def _terminal_dict(frame: pd.DataFrame) -> dict[str, float | None]:
    return {key: _finite_number(value) for key, value in frame.iloc[-1].items()}


def _fetch_native(symbol: str, interval: str, lookback_days: int) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """Fetch independently with the same start/end style as production load_eod."""
    started = datetime.now(timezone.utc)
    end = pd.Timestamp(started)
    start = end - timedelta(days=lookback_days)
    ticker = yf.Ticker(symbol)  # fresh object prevents cross-interval metadata bleed
    raw = ticker.history(
        interval=interval,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        prepost=False,
    )
    completed = datetime.now(timezone.utc)
    try:
        metadata = ticker.get_history_metadata() or {}
    except Exception as exc:  # metadata is evidence, not a reason to discard valid bars
        metadata = {"metadata_error": f"{type(exc).__name__}: {exc}"}
    timing = {
        "fetch_started_at_utc": iso_utc(started),
        "fetch_completed_at_utc": iso_utc(completed),
        "fetch_started_at_et": started.astimezone(ET).isoformat(),
        "fetch_completed_at_et": completed.astimezone(ET).isoformat(),
    }
    return canonicalize(raw), json_safe(metadata), timing


def compare_symbol(symbol: str, *, lookback_days: int = 400) -> dict[str, Any]:
    frames: dict[str, pd.DataFrame] = {}
    result: dict[str, Any] = {
        "symbol": symbol,
        "provider": "Yahoo via yfinance",
        "request": {
            "start_end_lookback_days": lookback_days,
            "auto_adjust": False,
            "actions": False,
            "prepost": False,
            "session_semantics": "regular-hours EOD (production-compatible)",
        },
        "intervals": {},
    }
    for interval in INTERVALS:
        try:
            frame, metadata, timing = _fetch_native(symbol, interval, lookback_days)
        except Exception as exc:
            result["intervals"][interval] = {"status": "fetch_error", "error": f"{type(exc).__name__}: {exc}"}
            continue
        frames[interval] = frame
        entry: dict[str, Any] = {**timing, "provider_metadata": metadata}
        if frame.empty:
            entry.update(status="empty", row_count=0)
        else:
            entry.update(
                status="ok",
                row_count=len(frame),
                interval_label=frame.index[-1].isoformat(),
                terminal=_terminal_dict(frame),
                penultimate=(
                    {"interval_label": frame.index[-2].isoformat(), **_terminal_dict(frame.iloc[-2: -1])}
                    if len(frame) > 1 else None
                ),
                payload_sha256=payload_hash(frame),
                price_tolerance=price_tolerance(metadata),
            )
        result["intervals"][interval] = entry

    daily = frames.get("1d")
    if daily is not None and not daily.empty:
        result["latest_daily_market_date"] = daily.index[-1].date().isoformat()
        for interval in ("1wk", "1mo"):
            entry = result["intervals"].get(interval, {})
            if entry.get("status") != "ok":
                continue
            derived = {key: _finite_number(value) for key, value in derive_terminal(daily, interval).items()}
            entry["daily_derived_secondary"] = derived
            entry["completion_corroboration"] = corroborates_completion(
                entry["terminal"], derived, entry["price_tolerance"]
            )
            entry["completion_evidence_note"] = (
                "Native O/H/L/V matches same-run daily aggregation; Close excluded. This corroborates, "
                "but provider metadata does not prove, completion through the daily terminal session."
            )
    closes = {
        interval: result["intervals"].get(interval, {}).get("terminal", {}).get("close")
        for interval in INTERVALS
    }
    result["close_differences"] = close_differences(closes)
    result["assessment"] = classify_observation(result["intervals"])
    return result


def _flatten_observation(observation: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terminal_rows: list[dict[str, Any]] = []
    symbol = observation["symbol"]
    for interval in INTERVALS:
        entry = observation["intervals"].get(interval, {})
        terminal_rows.append(
            {
                "observation": observation["observation"], "symbol": symbol,
                "requested_interval": interval, "status": entry.get("status"),
                "interval_label": entry.get("interval_label"), **entry.get("terminal", {}),
                "provider_timezone": entry.get("provider_metadata", {}).get("exchangeTimezoneName"),
                "exchange": entry.get("provider_metadata", {}).get("exchangeName"),
                "instrument_type": entry.get("provider_metadata", {}).get("instrumentType"),
                "fetch_started_at_utc": entry.get("fetch_started_at_utc"),
                "fetch_started_at_et": entry.get("fetch_started_at_et"),
                "completion_corroboration": entry.get("completion_corroboration"),
                "payload_sha256": entry.get("payload_sha256"),
            }
        )
    comparison = {
        "observation": observation["observation"], "symbol": symbol,
        "latest_daily_market_date": observation.get("latest_daily_market_date"),
        "classification": observation["assessment"]["classification"],
        "comparable": observation["assessment"]["comparable"],
        **{f"close_{i}": observation["intervals"].get(i, {}).get("terminal", {}).get("close") for i in INTERVALS},
    }
    for pair, values in observation["close_differences"].items():
        comparison[f"{pair}_absolute"] = values["absolute"]
        comparison[f"{pair}_relative"] = values["relative"]
    return terminal_rows, comparison


def _report(bundle: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for item in bundle["observations"]:
        label = item["assessment"]["classification"]
        counts[label] = counts.get(label, 0) + 1
    lines = [
        "# Yahoo native D/W/M terminal-close diagnostic", "",
        f"Generated: `{bundle['generated_at_utc']}`", "",
        "## OBSERVED", "",
        f"* Independent native `1d`, `1wk`, and `1mo` Yahoo requests were made for: {', '.join(bundle['symbols'])}.",
        "* Requests explicitly used `auto_adjust=False`, `actions=False`, and `prepost=False`.",
        "* Classification counts: " + json.dumps(counts, sort_keys=True) + ".",
        "* Exact rows, request timestamps, hashes, penultimate rows, and provider metadata are in `raw_native_observations.json`; tabular terminal rows are in `native_terminal_rows.csv`.", "",
        "## DERIVED", "",
        "* `daily_derived_secondary` aggregates the current week/month from native daily data. It is secondary evidence, never the native W/M result.",
        "* W/M completion corroboration compares Open/High/Low/Volume only and deliberately excludes Close, avoiding circular use of the invariant under test.", "",
        "## INFERRED", "",
        "* A daily terminal label is treated as the latest incorporated daily session for this diagnostic. Whether that session is final is assessed from fetch time and Yahoo trading-period metadata by the reviewer.",
        "* Matching native W/M O/H/L/V to the same-run daily aggregate is strong corroboration that they share the daily information set, but is not a provider guarantee.",
        "* `priceHint` supplies a deterministic display-precision tolerance (half one displayed unit); exact float equality is reported separately.", "",
        "## NOT PROVEN", "",
        "* Yahoo's native W/M bucket-start labels and history metadata do not state the last trading session incorporated into a partial aggregate.",
        "* `regularMarketTime`/`currentTradingPeriod` describe quote/market state, not per-bar W/M completion. Same-run timing alone is not proof.",
        "* One run cannot establish asynchronous update behavior, boundary behavior, or corporate-action revision behavior. Compare observations and rerun on the listed calendar positions.",
        "* `auto_adjust=False` plus separate Adj Close establishes request/response provenance, but `actions=False` suppresses action rows; it cannot prove absence or timing of upstream revisions.", "",
        "## Calendar position", "",
        f"* The workflow execution date directly observed here is `{bundle['generated_at_et'][:10]}` ET. Interpret its weekday/month-boundary/short-session status from the captured date; no historical partial payload is fabricated.",
        "* Rerun after close on a first session of week, midweek, Friday, first session of month, week crossing month-end, month-end, and (if practical) a shortened session.", "",
        "## Decision gate", "",
        "* **A — generic design:** repeated after-close observations across all target calendar positions show non-Close completion corroboration and exact/tolerance D/W/M closes for every well-formed symbol, with stable behavior across spaced observations and revisions.",
        "* **B — limited design:** agreement is reliable only for identifiable exchanges/times/calendar states, or completion can only be corroborated under stricter gates. Reconcile only inside those gates and fail closed otherwise.",
        "* **C — reject recovery:** W/M completion remains unidentifiable, corroborated same-information-set payloads disagree materially, or asynchronous/revision states cannot be deterministically excluded.",
        "* Before production work, gather repeated runs shortly after close and later the same evening, plus the calendar positions above and representative split/dividend events. Preserve provider/yfinance versions and full provenance.", "",
    ]
    return "\n".join(lines)


def write_artifacts(bundle: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_native_observations.json").write_text(json.dumps(bundle, indent=2, default=json_safe) + "\n")
    terminal_rows, comparisons = [], []
    metadata: list[dict[str, Any]] = []
    for item in bundle["observations"]:
        rows, comparison = _flatten_observation(item)
        terminal_rows.extend(rows)
        comparisons.append(comparison)
        for interval in INTERVALS:
            metadata.append({
                "observation": item["observation"], "symbol": item["symbol"],
                "requested_interval": interval,
                "provider_metadata": item["intervals"].get(interval, {}).get("provider_metadata"),
            })
    pd.DataFrame(terminal_rows).to_csv(output_dir / "native_terminal_rows.csv", index=False)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "normalized_comparisons.csv", index=False)
    summary = {"generated_at_utc": bundle["generated_at_utc"], "rows": comparisons}
    (output_dir / "agreement_summary.json").write_text(json.dumps(summary, indent=2, default=json_safe) + "\n")
    (output_dir / "provider_metadata.json").write_text(json.dumps(metadata, indent=2, default=json_safe) + "\n")
    (output_dir / "report.md").write_text(_report(bundle))


def parse_symbols(value: str) -> list[str]:
    symbols = [item.strip().upper() for item in value.replace(",", " ").split() if item.strip()]
    if not symbols or any(not all(char.isalnum() or char in ".-^=" for char in symbol) for symbol in symbols):
        raise ValueError("symbols must be non-empty Yahoo ticker tokens")
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=" ".join(DEFAULT_SYMBOLS), help="space/comma-separated symbols")
    parser.add_argument("--lookback-days", type=int, default=400)
    parser.add_argument("--observations", type=int, default=2)
    parser.add_argument("--delay-seconds", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/yahoo-native-dwm"))
    args = parser.parse_args()
    if not 30 <= args.lookback_days <= 3650:
        parser.error("--lookback-days must be between 30 and 3650")
    if not 1 <= args.observations <= 6 or not 0 <= args.delay_seconds <= 1800:
        parser.error("observations must be 1..6 and delay-seconds 0..1800")
    try:
        symbols = parse_symbols(args.symbols)
    except ValueError as exc:
        parser.error(str(exc))

    generated = datetime.now(timezone.utc)
    bundle: dict[str, Any] = {
        "schema_version": 2,
        "evidence_kind": "current_direct_provider_observation",
        "generated_at_utc": iso_utc(generated),
        "generated_at_et": generated.astimezone(ET).isoformat(),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "symbols": symbols,
        "observations": [],
    }
    for number in range(1, args.observations + 1):
        for symbol in symbols:
            item = compare_symbol(symbol, lookback_days=args.lookback_days)
            item["observation"] = number
            bundle["observations"].append(item)
        # Write incrementally so a later network failure still leaves auditable evidence.
        write_artifacts(bundle, args.output_dir)
        if number < args.observations:
            time.sleep(args.delay_seconds)
    print(args.output_dir)


if __name__ == "__main__":
    main()
