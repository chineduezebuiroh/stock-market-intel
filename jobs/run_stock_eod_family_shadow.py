"""Manual, non-authoritative stock EOD D/W/M family diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import storage
from etl.eod_family import (
    FAMILY_INTERVALS,
    MANIFEST_SCHEMA,
    TIMEFRAME_BY_INTERVAL,
    acquire_stock_eod_family,
    family_manifest,
    reconcile_stock_eod_family,
    shadow_manifest_path,
    write_shadow_manifest,
)
from etl.universe import symbols_for_universe
from indicators.core import apply_core, initialize_indicator_engine


LIVE_PATTERN_SYMBOLS = ("AAPL", "MSFT", "SPY", "CRWD", "NVDA", "JPM", "XOM")


def _symbols(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    if args.symbols:
        values.extend(args.symbols)
    if args.live_pattern:
        values.extend(LIVE_PATTERN_SYMBOLS)
    if args.universe:
        values.extend(symbols_for_universe(args.universe))
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _run_summary(run_id: str, manifests: list[pd.DataFrame]) -> pd.DataFrame:
    row = {column: None for column in MANIFEST_SCHEMA}
    interval_rows = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    if interval_rows.empty:
        first_fetch = last_fetch = None
        duration = None
    else:
        starts = pd.to_datetime(interval_rows["fetch_started_at"], utc=True)
        completions = pd.to_datetime(interval_rows["fetch_completed_at"], utc=True)
        first_fetch = starts.min()
        last_fetch = completions.max()
        duration = (last_fetch - first_fetch).total_seconds()
    row.update(
        record_type="run_summary", family_run_id=run_id,
        acquisition_status="complete",
        fetch_started_at=first_fetch.isoformat() if first_fetch is not None else None,
        fetch_completed_at=last_fetch.isoformat() if last_fetch is not None else None,
        family_duration_seconds=duration,
        requests_attempted=int(interval_rows.drop_duplicates(["family_run_id", "symbol"])
                               ["requests_attempted"].sum()) if not interval_rows.empty else 0,
        requests_succeeded=int(interval_rows.drop_duplicates(["family_run_id", "symbol"])
                               ["requests_succeeded"].sum()) if not interval_rows.empty else 0,
        requests_failed=int(interval_rows.drop_duplicates(["family_run_id", "symbol"])
                            ["requests_failed"].sum()) if not interval_rows.empty else 0,
        requested_intervals=",".join(FAMILY_INTERVALS),
        reconciliation_provider_requests=0,
    )
    return pd.DataFrame([row], columns=MANIFEST_SCHEMA)


def run_shadow(symbols: list[str], run_id: str, *, process_indicators: bool = True) -> tuple[Path, Path | None]:
    """Acquire and finalize families; write only beneath data/_shadow/eod_family."""
    if not symbols:
        raise ValueError("at least one symbol is required")
    manifests: list[pd.DataFrame] = []
    indicator_rows: list[pd.DataFrame] = []
    if process_indicators:
        initialize_indicator_engine()
    for symbol in symbols:
        family = acquire_stock_eod_family(run_id, symbol)
        finalized = reconcile_stock_eod_family(family)
        manifests.append(family_manifest(finalized))
        if process_indicators:
            for interval, observation in zip(
                FAMILY_INTERVALS, (finalized.daily, finalized.weekly, finalized.monthly)
            ):
                if (observation is None or observation.finalized_frame is None
                        or observation.data_quality_status not in {"valid", "repaired"}):
                    continue
                processed = apply_core(
                    observation.finalized_frame.copy(deep=True), namespace="stocks",
                    timeframe=TIMEFRAME_BY_INTERVAL[interval],
                )
                if processed is not None and not processed.empty:
                    terminal = processed.tail(1).copy()
                    terminal.insert(0, "interval", interval)
                    terminal.insert(0, "symbol", symbol)
                    terminal.insert(0, "family_run_id", run_id)
                    terminal.insert(3, "terminal_label", terminal.index.astype(str))
                    terminal = terminal.reset_index(drop=True)
                    indicator_rows.append(terminal)
    manifest = pd.concat(manifests + [_run_summary(run_id, manifests)],
                         ignore_index=True)
    manifest_path = write_shadow_manifest(manifest, run_id)
    indicator_path = None
    if indicator_rows:
        indicator_path = shadow_manifest_path(run_id).with_name("indicator_terminals.parquet")
        storage.save_parquet(pd.concat(indicator_rows, ignore_index=True), indicator_path)
    return manifest_path, indicator_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", help="tiny explicit symbol set")
    parser.add_argument("--live-pattern", action="store_true", help="the established seven symbols")
    parser.add_argument("--universe", choices=("shortlist_stocks", "options_eligible"))
    parser.add_argument("--run-id", default=f"shadow-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--no-indicators", action="store_true")
    args = parser.parse_args()
    symbols = _symbols(args)
    path, indicators = run_shadow(symbols, args.run_id, process_indicators=not args.no_indicators)
    print(f"[SHADOW] manifest={path}")
    if indicators:
        print(f"[SHADOW] indicator_terminals={indicators}")
    print("[SHADOW] no canonical snapshots, combos, notifications, or registry state were touched")


if __name__ == "__main__":
    main()
