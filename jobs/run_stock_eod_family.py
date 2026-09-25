"""Authoritative production runner for stock daily/weekly/monthly EOD families."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import uuid

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import storage
from core.paths import DATA, CFG
from etl.eod_family import (
    FAMILY_INTERVALS,
    TIMEFRAME_BY_INTERVAL,
    acquire_stock_eod_family,
    family_manifest,
    frame_sha256,
    reconcile_stock_eod_family,
)
from etl.sources import safe_load_eod
from indicators.composite_spy_qqq_volume_ma_ratio import (
    REFERENCE_EOD_WINDOW_BARS,
    REFERENCE_SYMBOLS,
    preloaded_eod_references,
)
from indicators.core import initialize_indicator_engine
from jobs.run_timeframe import (
    TF_CFG,
    build_timeframe_snapshot,
    process_one_preloaded,
    symbols_for_timeframe,
)


TIMEFRAMES = tuple(TIMEFRAME_BY_INTERVAL[interval] for interval in FAMILY_INTERVALS)


def resolve_family_symbols() -> list[str]:
    """Require independently resolved production D/W/M universes to be identical."""
    resolved = {
        timeframe: set(symbols_for_timeframe("stocks", timeframe))
        for timeframe in TIMEFRAMES
    }
    baseline = resolved["daily"]
    if any(symbols != baseline for symbols in resolved.values()):
        counts = ", ".join(f"{name}={len(symbols)}" for name, symbols in resolved.items())
        differences = []
        for name in ("weekly", "monthly"):
            missing = sorted(baseline - resolved[name])[:10]
            extra = sorted(resolved[name] - baseline)[:10]
            if missing or extra:
                differences.append(f"{name}: missing_vs_daily={missing}, extra_vs_daily={extra}")
        raise RuntimeError(
            "stock EOD family universe mismatch; refusing acquisition: "
            f"{counts}; {'; '.join(differences)}"
        )
    if not baseline:
        raise RuntimeError("stock EOD family universe is empty")
    return sorted(baseline)


def production_window_bars() -> dict[str, int]:
    """Resolve acquisition windows from the authoritative timeframe config."""
    return {
        interval: int(TF_CFG["stocks"][TIMEFRAME_BY_INTERVAL[interval]]["window_bars"])
        for interval in FAMILY_INTERVALS
    }


def production_manifest_path(run_id: str) -> Path:
    return DATA / "_health" / "stocks_eod_family" / run_id / "manifest.parquet"


def acquire_reference_data(
    *,
    session: str,
    loader: Callable = safe_load_eod,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame, int, int, int]:
    """Acquire the bounded SPY/QQQ D/W/M indicator inputs exactly once."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    attempted = succeeded = failed = 0
    for symbol in REFERENCE_SYMBOLS:
        for interval in FAMILY_INTERVALS:
            timeframe = TIMEFRAME_BY_INTERVAL[interval]
            attempted += 1
            frame = None
            error = None
            try:
                frame = loader(
                    symbol,
                    timeframe=timeframe,
                    window_bars=REFERENCE_EOD_WINDOW_BARS,
                    session=session,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            valid = (
                frame is not None
                and not frame.empty
                and "volume" in frame.columns
                and pd.to_numeric(frame["volume"], errors="coerce").notna().any()
            )
            if valid:
                frames[(symbol, timeframe)] = frame.copy(deep=True)
                succeeded += 1
            else:
                failed += 1
                error = error or "empty or invalid benchmark volume history"
            rows.append({
                "record_type": "reference",
                "symbol": symbol,
                "interval": interval,
                "timeframe": timeframe,
                "acquisition_status": "success" if valid else "failed",
                "acquisition_error": error,
                "payload_sha256": frame_sha256(frame),
                "requests_attempted": 1,
                "requests_succeeded": int(valid),
                "requests_failed": int(not valid),
                "reconciliation_provider_requests": 0,
            })
    if failed:
        failures = [
            f"{row['symbol']}:{row['interval']}={row['acquisition_error']}"
            for row in rows if row["acquisition_status"] == "failed"
        ]
        raise RuntimeError(
            "stock EOD reference-data acquisition failed: " + "; ".join(failures)
        )
    return frames, pd.DataFrame(rows), attempted, succeeded, failed


def _summary_row(
    run_id: str,
    symbol_count: int,
    attempted: int,
    succeeded: int,
    failed: int,
    reconciliation_requests: int,
    reference_attempted: int,
    reference_succeeded: int,
    reference_failed: int,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "record_type": "run_summary",
        "family_run_id": run_id,
        "routed_symbol_count": symbol_count,
        "requests_attempted": attempted,
        "requests_succeeded": succeeded,
        "requests_failed": failed,
        "requested_intervals": ",".join(FAMILY_INTERVALS),
        "reconciliation_provider_requests": reconciliation_requests,
        "primary_provider_attempts": attempted,
        "reference_provider_attempts": reference_attempted,
        "reference_provider_successes": reference_succeeded,
        "reference_provider_failures": reference_failed,
        "total_provider_attempts": attempted + reference_attempted,
        "downstream_provider_requests": 0,
    }])


def run_family(
    *,
    run_id: str | None = None,
    acquire: Callable | None = None,
    reference_loader: Callable = safe_load_eod,
) -> Path:
    """Acquire each D/W/M family once and publish existing canonical products."""
    run_id = run_id or (
        f"prod-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    acquire = acquire or acquire_stock_eod_family
    symbols = resolve_family_symbols()
    windows = production_window_bars()
    sessions = {TF_CFG["stocks"][timeframe]["session"] for timeframe in TIMEFRAMES}
    if len(sessions) != 1:
        raise RuntimeError(f"stock EOD family session mismatch: {sorted(sessions)}")
    session = sessions.pop()

    initialize_indicator_engine(CFG)
    references, reference_manifest, reference_attempted, reference_succeeded, reference_failed = (
        acquire_reference_data(session=session, loader=reference_loader)
    )
    rejected = {timeframe: set() for timeframe in TIMEFRAMES}
    manifests: list[pd.DataFrame] = []
    attempted = succeeded = failed = reconciliation_requests = 0

    with preloaded_eod_references(references):
        for index, symbol in enumerate(symbols, start=1):
            family = acquire(
                run_id, symbol, session=session, window_bars=windows,
            )
            finalized = reconcile_stock_eod_family(family)
            attempted += family.requests.attempted
            succeeded += family.requests.succeeded
            failed += family.requests.failed
            reconciliation_requests += family.requests.reconciliation_provider_requests

            manifest = family_manifest(finalized)
            manifest["finalization_status"] = "excluded"
            manifest["processing_status"] = "not_processed"
            manifest["processing_reason"] = None

            observations = {
                "1d": finalized.daily,
                "1wk": finalized.weekly,
                "1mo": finalized.monthly,
            }
            for interval in FAMILY_INTERVALS:
                timeframe = TIMEFRAME_BY_INTERVAL[interval]
                observation = observations[interval]
                row_mask = manifest["interval"] == interval
                status = observation.data_quality_status if observation is not None else "excluded"
                manifest.loc[row_mask, "finalization_status"] = status
                if (
                    observation is None
                    or observation.finalized_frame is None
                    or status not in {"valid", "repaired"}
                ):
                    rejected[timeframe].add(symbol)
                    manifest.loc[row_mask, "processing_status"] = "excluded"
                    manifest.loc[row_mask, "processing_reason"] = "finalized observation is not eligible"
                    continue
                outcome = process_one_preloaded(
                    "stocks", timeframe, symbol, observation.finalized_frame,
                    windows[interval], progress_index=index, progress_total=len(symbols),
                )
                if not outcome.accepted:
                    rejected[timeframe].add(symbol)
                    manifest.loc[row_mask, "processing_status"] = "rejected"
                    manifest.loc[row_mask, "processing_reason"] = outcome.reason
                else:
                    manifest.loc[row_mask, "processing_status"] = "accepted"
            manifests.append(manifest)

    summary = _summary_row(
        run_id, len(symbols), attempted, succeeded, failed, reconciliation_requests,
        reference_attempted, reference_succeeded, reference_failed,
    )
    reference_manifest["family_run_id"] = run_id
    telemetry = pd.concat(
        manifests + [reference_manifest, summary], ignore_index=True, sort=False
    )
    path = production_manifest_path(run_id)
    storage.save_parquet(telemetry, path)
    print(f"[HEALTH] wrote stock EOD family telemetry: {path}", flush=True)

    expected_attempts = 3 * len(symbols)
    if attempted != expected_attempts:
        raise RuntimeError(
            f"stock EOD family request invariant failed: attempted={attempted}, "
            f"expected={expected_attempts}"
        )
    if reconciliation_requests != 0:
        raise RuntimeError(
            "stock EOD family reconciliation request invariant failed: "
            f"requests={reconciliation_requests}"
        )
    expected_reference_attempts = len(REFERENCE_SYMBOLS) * len(FAMILY_INTERVALS)
    if reference_attempted != expected_reference_attempts:
        raise RuntimeError(
            "stock EOD reference request invariant failed: "
            f"attempted={reference_attempted}, expected={expected_reference_attempts}"
        )

    for timeframe in TIMEFRAMES:
        build_timeframe_snapshot("stocks", timeframe, symbols, rejected[timeframe])
    return path


def main() -> None:
    path = run_family()
    print(f"[OK] Stock EOD family production run complete: {path}")


if __name__ == "__main__":
    main()
