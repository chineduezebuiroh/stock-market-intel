from __future__ import annotations

# jobs/run_timeframe.py

import sys
import math
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import DATA, CFG  # no REF
from core import storage

import pandas as pd
import yaml

from etl.sources import (
    load_eod,
    load_130m_from_5m,
    load_stocks_intraday_4h_extended,
    load_quarterly_from_monthly,
    load_yearly_from_monthly,
    load_futures_intraday,
    safe_load_eod,
    INTRADAY_REPAIR_TELEMETRY,
)
from etl.window import parquet_path, update_fixed_window
from etl.eod_quality import (
    StockEODDataQualityError,
    merge_stock_eod_window,
    require_valid_terminal_stock_eod,
    valid_stock_eod_rows,
)
from etl.eod_reconciliation import (
    canonical_terminal_session_close,
    observation_from_native_frame,
    repair_terminal_close,
)
from etl.universe import symbols_for_universe
#from etl.futures_resample import load_futures_eod_from_1h 
from etl.futures_resample import load_futures_eod_hybrid

from functools import lru_cache

from indicators.core import (
    apply_core,
    get_snapshot_base_cols,
    initialize_indicator_engine,
)

import time

import inspect
from dataclasses import dataclass
print("[DEBUG] load_eod signature:", inspect.signature(load_eod))


DEV_MAX_STOCK_SYMBOLS_PER_TF = None  # set to None to disable the cap
EXCLUSIONS_FILE = CFG / "excluded_symbols.csv"
EOD_RECONCILIATION_TELEMETRY: list[dict] = []
EOD_RECONCILIATION_ATTEMPTS = 0
EOD_RECONCILIATION_ELIGIBLE = 0
EOD_RECONCILIATION_SKIPPED = 0
MAX_EOD_RECONCILIATION_SYMBOLS = int(
    os.getenv("MAX_EOD_RECONCILIATION_SYMBOLS", "100")
)
EOD_RECONCILIATION_SELECTION_POLICY = "daily_rotating_sha256_utc_date"


def _price_tolerance(frame: pd.DataFrame) -> float:
    metadata = frame.attrs.get("eod_provenance", {}).get("provider_metadata", {})
    try:
        hint = int(metadata.get("priceHint", 4))
        if not 0 <= hint <= 12:
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        hint = 4
    return max(1e-12, 0.5 * 10 ** (-hint))


def _is_close_only_malformed(frame: pd.DataFrame) -> bool:
    if frame is None or frame.empty:
        return False
    row = frame.iloc[-1]
    close = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
    fields = {c: pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
              for c in ("open", "high", "low", "volume")}
    return (
        pd.isna(close)
        and all(math.isfinite(value) for value in fields.values())
        and fields["volume"] >= 0
        and fields["low"] <= fields["high"]
    )


def _reconcile_malformed_daily_close(
    symbol: str, frame: pd.DataFrame, *, session: str
) -> tuple[pd.DataFrame, dict]:
    """Narrow production gate for the evidenced malformed-D/valid-W+M state."""
    if not _is_close_only_malformed(frame):
        return frame, {"data_quality_status": "malformed", "reason": "not close-only malformed"}

    peers = []
    source_statuses = {"1d": "malformed"}
    for peer, bars in (("weekly", 8), ("monthly", 3)):
        try:
            fetched = safe_load_eod(
                symbol, timeframe=peer, window_bars=bars, session=session
            )
        except Exception as exc:
            # safe_load_eod already contains timeout/error isolation; keep this
            # defensive boundary so a future/custom loader remains symbol-local.
            print(
                f"[DATA_QUALITY][EVIDENCE_FETCH_WARN] {symbol} {peer}: "
                f"{type(exc).__name__}: {exc}", flush=True,
            )
            fetched = None
        if fetched is not None and not fetched.empty:
            peers.append(fetched)
            source_statuses["1wk" if peer == "weekly" else "1mo"] = (
                "valid" if pd.notna(pd.to_numeric(
                    pd.Series([fetched.iloc[-1].get("close")]), errors="coerce"
                ).iloc[0]) else "malformed"
            )
        else:
            source_statuses["1wk" if peer == "weekly" else "1mo"] = "unavailable"
    target_session = pd.Timestamp(frame.index[-1]).date()
    observations = [observation_from_native_frame(peer) for peer in peers]
    tolerance = max((_price_tolerance(peer) for peer in peers), default=1e-4)
    result = canonical_terminal_session_close(
        observations, target_session=target_session, price_tolerance=tolerance
    )
    repaired, provenance = repair_terminal_close(frame, result)
    provenance.update(
        symbol=symbol, target_interval="1d", interval_label=str(frame.index[-1]),
        terminal_market_session_date=target_session.isoformat(),
        terminal_close_pattern="/".join(
            f"{interval}:{source_statuses[interval]}" for interval in ("1d", "1wk", "1mo")
        ),
    )
    return repaired, provenance


def write_eod_reconciliation_telemetry(timeframe: str) -> None:
    if not EOD_RECONCILIATION_TELEMETRY:
        return
    out = DATA / "_health" / f"stocks_{timeframe}_eod_reconciliation.parquet"
    rows = list(EOD_RECONCILIATION_TELEMETRY)
    rows.append({
        "record_type": "run_summary",
        "selection_policy": EOD_RECONCILIATION_SELECTION_POLICY,
        "eligible_count": EOD_RECONCILIATION_ELIGIBLE,
        "attempted_count": EOD_RECONCILIATION_ATTEMPTS,
        "skipped_budget_count": EOD_RECONCILIATION_SKIPPED,
        "budget": MAX_EOD_RECONCILIATION_SYMBOLS,
    })
    try:
        storage.save_parquet(pd.DataFrame(rows), out)
    except Exception as exc:
        print(
            f"[HEALTH][WARN] EOD reconciliation telemetry write failed for {out}: "
            f"{type(exc).__name__}: {exc}", flush=True,
        )
        return
    print(f"[HEALTH] wrote EOD reconciliation telemetry: {out}", flush=True)


# Load timeframe config (structure only)
with open(CFG / "timeframes.yaml", "r") as f:
    TF_CFG = yaml.safe_load(f)

# Load multi-timeframe combos (for universes per timeframe)
with open(CFG / "multi_timeframe_combos.yaml", "r") as f:
    MTF_CFG = yaml.safe_load(f)

CASCADE = {
    "stocks": {
        "intraday_130m": ["daily", "weekly"],
        "intraday_4h": ["daily", "weekly"],
        "daily": ["weekly", "monthly"],
        "weekly": ["monthly", "quarterly"],
        "monthly": ["quarterly", "yearly"],
    },
    "futures": {
        "intraday_1h": ["intraday_4h", "daily"],
        "intraday_4h": ["daily", "weekly"],
        "daily": ["weekly", "monthly"],
    },
}


@lru_cache(maxsize=1)
def _load_symbol_exclusions() -> set[str]:
    """
    Load a global list of symbols to exclude from all timeframes/universes.
    Expected file: config/excluded_symbols.csv with at least a 'symbol' column.
    """
    if not EXCLUSIONS_FILE.exists():
        return set()

    try:
        df = pd.read_csv(EXCLUSIONS_FILE)
    except Exception as e:
        print(f"[WARN] Failed to read exclusions from {EXCLUSIONS_FILE}: {e}")
        return set()

    col = None
    for candidate in ("symbol", "Symbol", "ticker", "Ticker"):
        if candidate in df.columns:
            col = candidate
            break

    if col is None:
        print(
            f"[WARN] {EXCLUSIONS_FILE} has no 'symbol' or 'ticker' column; "
            "no exclusions will be applied."
        )
        return set()

    # Normalize to upper-case strings with no spaces
    symbols = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    return set(symbols)


def write_intraday_repair_telemetry(namespace: str, timeframe: str) -> None:
    if not INTRADAY_REPAIR_TELEMETRY:
        print(f"[HEALTH] no repair telemetry for {namespace}:{timeframe}")
        return

    rows = list(INTRADAY_REPAIR_TELEMETRY.values())
    df = pd.DataFrame(rows)

    out = DATA / "_health" / f"{namespace}_{timeframe}_repair_telemetry.parquet"
    storage.save_parquet(df, out)
    print(f"[HEALTH] wrote repair telemetry: {out}")


def universes_for_timeframe(namespace: str, timeframe: str) -> list[str]:
    """
    Look through all combos in multi_timeframe_combos.yaml and find which
    universes use this namespace+timeframe in any role (lower/middle/upper).
    """
    ns_cfg = MTF_CFG.get(namespace, {})
    universes = set()

    for combo_name, combo_cfg in ns_cfg.items():
        if combo_cfg.get("lower_tf") == timeframe:
            universes.add(combo_cfg["universe"])
        if combo_cfg.get("middle_tf") == timeframe:
            universes.add(combo_cfg["universe"])
        if combo_cfg.get("upper_tf") == timeframe:
            universes.add(combo_cfg["universe"])

    return sorted(universes)


def symbols_for_timeframe(namespace: str, timeframe: str, allowed_universes: set[str] | None = None) -> list[str]:
    """
    Union of all symbols for all universes that reference this timeframe,
    with a dev-only cap for stocks so local runs stay fast.
    """
    universes = universes_for_timeframe(namespace, timeframe)

    if allowed_universes is not None:
        universes = [u for u in universes if u in allowed_universes]

    shortlist_syms: set[str] = set()
    other_syms: set[str] = set()

    for u in universes:
        syms = set(symbols_for_universe(u))
        if u.startswith("shortlist_"):
            shortlist_syms.update(syms)
        else:
            other_syms.update(syms)

    # ---- normalize everything once (match exclusions normalization) ----
    def norm(x) -> str:
        return str(x).strip().upper()

    shortlist_syms = {norm(s) for s in shortlist_syms if s is not None and str(s).strip()}
    other_syms     = {norm(s) for s in other_syms if s is not None and str(s).strip()}

    excluded = _load_symbol_exclusions() or set()

    # ✅ apply exclusions ONLY to non-shortlist set
    other_syms = {s for s in other_syms if s not in excluded}

    # Base set: always include all shortlist symbols
    if namespace == "stocks" and DEV_MAX_STOCK_SYMBOLS_PER_TF is not None:
        # Remaining slots after we include all shortlist symbols
        remaining = DEV_MAX_STOCK_SYMBOLS_PER_TF - len(shortlist_syms)
        if remaining <= 0:
            # Dev cap fully consumed by shortlist; just return them.
            symbols = sorted(shortlist_syms)
        else:
            # Only consider non-shortlist symbols as extras
            extra_candidates = sorted(other_syms - shortlist_syms)
            extra = extra_candidates[:remaining]
            symbols = sorted(shortlist_syms.union(extra))
    else:
        # futures or no dev cap: everything
        symbols = sorted(shortlist_syms.union(other_syms))

    return symbols


@dataclass(frozen=True)
class ProcessOutcome:
    accepted: bool
    reason: str | None = None


def process_one_preloaded(
    namespace: str,
    timeframe: str,
    symbol: str,
    df_new: pd.DataFrame,
    window_bars: int,
    *,
    progress_index: int | None = None,
    progress_total: int | None = None,
) -> ProcessOutcome:
    """Merge, validate, calculate, and persist an already-acquired frame.

    This is deliberately below the provider boundary: it never loads market data.
    Both standalone ingestion and the authoritative D/W/M family runner use this
    function so quality, rolling-window, indicator, and persistence semantics stay
    identical.
    """
    stock_eod = namespace == "stocks" and timeframe in {"daily", "weekly", "monthly"}
    parquet = parquet_path(DATA, f"{namespace}_{timeframe}", symbol)
    parquet.parent.mkdir(parents=True, exist_ok=True)

    if storage.exists(parquet):
        existing = storage.load_parquet(parquet)
        bad_cols = [
            c for c in existing.columns
            if isinstance(c, str) and c.startswith("(")
        ]
        if bad_cols:
            existing = existing.drop(columns=bad_cols)
    else:
        existing = pd.DataFrame()

    if stock_eod:
        try:
            merge_result = merge_stock_eod_window(
                df_new, existing, window_bars,
                # A D label identifies its session. W/M labels identify only
                # a bucket and cannot prove an old partial row is current.
                allow_same_label_retention=(timeframe == "daily"),
            )
            merged = merge_result.frame
            if merge_result.dropped_malformed_timestamps:
                print(
                    f"[DATA_QUALITY][RETAIN_EXISTING] {namespace}:{timeframe} {symbol} "
                    f"timestamps={merge_result.retained_existing_timestamps} "
                    f"dropped={merge_result.dropped_malformed_timestamps}",
                    flush=True,
                )
        except StockEODDataQualityError as exc:
            print(f"[DATA_QUALITY][REJECT] {namespace}:{timeframe} {symbol}: {exc}", flush=True)
            return ProcessOutcome(False, str(exc))
    else:
        merged = update_fixed_window(df_new, existing, window_bars)

    if namespace == "futures" and timeframe in ("intraday_1h", "intraday_4h"):
        merged = merged.dropna(subset=["open", "high", "low", "close"], how="any")

    if merged is None or merged.empty:
        print(f"[INGEST][SKIP] {namespace}:{timeframe} {symbol} merged empty", flush=True)
        return ProcessOutcome(False, "merged frame is empty")

    merged = apply_core(merged, namespace=namespace, timeframe=timeframe)
    if merged is None or merged.empty:
        print(f"[INGEST][SKIP] {namespace}:{timeframe} {symbol} post-indicators empty", flush=True)
        return ProcessOutcome(False, "indicator output is empty")

    storage.save_parquet(merged, parquet)
    try:
        ok = storage.exists(parquet)
    except Exception as exc:
        ok = False
        print(f"[WRITE][ERR] exists-check failed for {parquet}: {exc}", flush=True)
    if not ok:
        print(f"[WRITE][MISS] {namespace}:{timeframe} {symbol} -> {parquet}", flush=True)
    elif progress_index is None or progress_index <= 3 or progress_index % 50 == 0:
        print(f"[WRITE][OK] {namespace}:{timeframe} {symbol} -> {parquet}", flush=True)

    if progress_index == 1:
        try:
            df_chk = storage.load_parquet(parquet)
            print(f"[WRITE][CHK] {namespace}:{timeframe} {symbol} saved shape={df_chk.shape}", flush=True)
        except Exception as exc:
            print(f"[WRITE][CHK][ERR] {namespace}:{timeframe} {symbol}: {exc}", flush=True)
    return ProcessOutcome(True)


def ingest_one(namespace: str, timeframe: str, symbols, session: str, window_bars: int):
    """
    Ingest bars for a single namespace+timeframe over a list of symbols,
    update fixed rolling window, apply indicators, and persist parquet.

    This version:
      - uses load_eod() with a start date derived from timeframe+window_bars,
      - relies on update_fixed_window() to enforce the sliding window.
      - drops any legacy pivot-style columns (e.g. "('open', 'aapl')")
        from existing data before merging.
    """
    global EOD_RECONCILIATION_ATTEMPTS, EOD_RECONCILIATION_ELIGIBLE
    global EOD_RECONCILIATION_SKIPPED
    total = len(symbols)
    print(f"[INGEST] {namespace}:{timeframe} starting ingest for {total} symbols", flush=True)

    

    rejected_symbols: set[str] = set()
    stock_eod = namespace == "stocks" and timeframe in {"daily", "weekly", "monthly"}

    selection_date = datetime.now(timezone.utc).date().isoformat()
    if namespace == "stocks" and timeframe == "daily":
        symbols = sorted(
            symbols,
            key=lambda symbol: hashlib.sha256(
                f"{selection_date}|{symbol}".encode()
            ).hexdigest(),
        )

    for idx, sym in enumerate(symbols, start=1):
        start_sym = time.perf_counter()
        print(f"[INGEST] {namespace}:{timeframe} [{idx}/{total}] {sym} ...")

        try:
            if namespace == "stocks" and timeframe == "intraday_130m":
                df_new = load_130m_from_5m(sym, session=session)

            elif namespace == "stocks" and timeframe == "intraday_4h":
                df_new = load_stocks_intraday_4h_extended(sym, window_bars=window_bars, session=session)
                
            elif namespace == "stocks" and timeframe == "yearly":
                df_new = load_yearly_from_monthly(sym, window_bars=window_bars, session=session)
    
            elif namespace == "stocks" and timeframe == "quarterly":
                df_new = load_quarterly_from_monthly(sym, window_bars=window_bars, session=session)
    
            elif namespace == "futures" and timeframe in ("intraday_1h", "intraday_4h"):
                df_new = load_futures_intraday(sym, timeframe=timeframe, window_bars=window_bars, session=session)
            
            # ✅ NEW: futures higher TFs derived from canonical 1h
            elif namespace == "futures" and timeframe in ("daily", "weekly", "monthly"):
                #df_new = load_futures_eod_from_1h(sym, timeframe=timeframe, window_bars=window_bars)
                df_new = load_futures_eod_hybrid(sym, timeframe=timeframe, window_bars=window_bars, session=session, vendor_loader=safe_load_eod)
                
            else:
                df_new = safe_load_eod(sym, timeframe=timeframe, window_bars=window_bars, session=session)
        
        except Exception as e:
            elapsed = time.perf_counter() - start_sym
            print(f"[INGEST][WARN] {namespace}:{timeframe} {sym} load exception after {elapsed:.1f}s: {e}", flush=True)
            if stock_eod:
                rejected_symbols.add(sym)
            continue

        # 🔹 NEW: explicitly log empty df_new
        if df_new is None or df_new.empty:
            elapsed = time.perf_counter() - start_sym
            print(f"[INGEST][SKIP] {namespace}:{timeframe} {sym} df_new empty/None after {elapsed:.1f}s", flush=True)
            if stock_eod:
                rejected_symbols.add(sym)
                print(f"[DATA_QUALITY][REJECT] {namespace}:{timeframe} {sym}: empty provider payload", flush=True)
            continue
        
        if stock_eod:
            # Reconciliation precedes PR #23's merge safety floor. Only the
            # evidenced daily close-only failure is activated for standalone runs.
            if timeframe == "daily" and _is_close_only_malformed(df_new):
                EOD_RECONCILIATION_ELIGIBLE += 1
                priority = hashlib.sha256(f"{selection_date}|{sym}".encode()).hexdigest()
                if EOD_RECONCILIATION_ATTEMPTS >= MAX_EOD_RECONCILIATION_SYMBOLS:
                    EOD_RECONCILIATION_SKIPPED += 1
                    provenance = {
                        "symbol": sym, "target_interval": "1d",
                        "data_quality_status": "excluded",
                        "reason": "supplemental fetch budget exhausted",
                    }
                else:
                    EOD_RECONCILIATION_ATTEMPTS += 1
                    try:
                        df_new, provenance = _reconcile_malformed_daily_close(
                            sym, df_new, session=session
                        )
                    except StockEODDataQualityError as exc:
                        rejected_symbols.add(sym)
                        print(
                            f"[DATA_QUALITY][REJECT] {namespace}:{timeframe} {sym}: {exc}",
                            flush=True,
                        )
                        continue
                provenance.update(
                    record_type="symbol", selection_policy=EOD_RECONCILIATION_SELECTION_POLICY,
                    selection_date=selection_date, selection_priority_sha256=priority,
                )
                EOD_RECONCILIATION_TELEMETRY.append(provenance)

        outcome = process_one_preloaded(
            namespace, timeframe, sym, df_new, window_bars,
            progress_index=idx, progress_total=total,
        )
        if not outcome.accepted:
            if stock_eod:
                rejected_symbols.add(sym)
            continue


        elapsed = time.perf_counter() - start_sym
        print(f"[INGEST] {namespace}:{timeframe} [{idx}/{total}] {sym} OK in {elapsed:.1f}s", flush=True)

    if stock_eod and timeframe == "daily":
        write_eod_reconciliation_telemetry(timeframe)
    return rejected_symbols


def build_timeframe_snapshot(
    namespace: str,
    timeframe: str,
    symbols,
    rejected_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """Build and publish the canonical snapshot from accepted rolling frames."""
    rejected_symbols = rejected_symbols or set()
    base_cols = get_snapshot_base_cols(namespace, timeframe)
    rows = []
    for sym in symbols:
        if sym in rejected_symbols:
            continue
        p = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        if not storage.exists(p):
            continue
        df = storage.load_parquet(p)
        if df.empty:
            continue
        if namespace == "futures" and timeframe in ("intraday_1h", "intraday_4h"):
            df_valid = df.dropna(subset=["open", "high", "low", "close"], how="any")
            if df_valid.empty:
                continue
            last = df_valid.iloc[-1]
        else:
            last = df.iloc[-1]
        if namespace == "stocks" and timeframe in {"daily", "weekly", "monthly"}:
            try:
                require_valid_terminal_stock_eod(df, context=f"snapshot {namespace}/{timeframe}/{sym}")
            except StockEODDataQualityError as exc:
                print(f"[DATA_QUALITY][SNAPSHOT_EXCLUDE] {exc}", flush=True)
                continue
        missing = [c for c in base_cols if c not in last.index]
        if missing:
            raise KeyError(f"Snapshot missing columns {missing} for {namespace}/{timeframe}/{sym}")
        row = last[base_cols].copy()
        row["symbol"] = sym
        rows.append(row)

    if rows:
        snap = pd.DataFrame(rows)[base_cols + ["symbol"]]
        if snap.index.name is None:
            snap.index.name = "date"
        snap.columns = snap.columns.astype(str)
        out = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
        storage.save_parquet(snap, out)
        print(f"[OK] Wrote snapshot: {out}")
        return snap
    if namespace == "stocks" and timeframe in {"daily", "weekly", "monthly"}:
        raise RuntimeError(
            f"[DATA_QUALITY][FATAL] no valid rows for {namespace}:{timeframe}; snapshot not published"
        )
    return pd.DataFrame()


def run(namespace: str, timeframe: str, cascade: bool = False, allowed_universes: set[str] | None = None):
    """
    Primary entry point: ingest for a namespace+timeframe for all symbols
    implied by the MTF combos, optionally cascade to higher timeframes,
    and build a single-timeframe snapshot.

    Snapshot behavior:
      - Always builds a snapshot_{namespace}_{timeframe}.parquet.
      - If a screen YAML exists, applies it via run_screen().
      - Otherwise, writes the raw latest-bar snapshot.
    """
    # --- 1) Determine config & symbols for this timeframe ---
    cfg_tf = TF_CFG[namespace][timeframe]
    session = cfg_tf["session"]
    window_bars = int(cfg_tf["window_bars"])

    symbols = symbols_for_timeframe(namespace, timeframe, allowed_universes)

    if not symbols:
        print(f"[WARN] No symbols found for {namespace}:{timeframe} via combos.")
        return

    # --- 2) Ingest this timeframe (per-symbol parquet with indicators) ---
    rejected_symbols = ingest_one(namespace, timeframe, symbols, session, window_bars)
    if namespace == "stocks" and timeframe == "intraday_4h":
        write_intraday_repair_telemetry(namespace, timeframe)

    # 3) Build single-timeframe snapshot (no screening/pivoting for now)
    build_timeframe_snapshot(namespace, timeframe, symbols, rejected_symbols)

    # --- 4) Cascade to higher timeframes (if requested) ---
    if cascade:
        child_tfs = CASCADE.get(namespace, {}).get(timeframe, [])
        for child_tf in child_tfs:
            # Important: child runs with cascade=False to avoid infinite recursion
            run(
                namespace,
                child_tf,
                cascade=False, # prevent infinite recursion
                allowed_universes=allowed_universes,  # propagate restriction
            )


if __name__ == "__main__":
    initialize_indicator_engine(CFG)

    if len(sys.argv) < 3:
        print("Usage: python jobs/run_timeframe.py <namespace> <timeframe> [--cascade] [--allowed-universes U1 U2 ...]")
        sys.exit(1)

    ns = sys.argv[1]
    tf = sys.argv[2]
    cascade = "--cascade" in sys.argv

    # Optional: --allowed-universes U1 U2 ...
    allowed_universes: set[str] | None = None
    if "--allowed-universes" in sys.argv:
        idx = sys.argv.index("--allowed-universes")
        universes: list[str] = []
        
        # collect args until the next flag or end of argv
        for arg in sys.argv[idx + 1 :]:
            if arg.startswith("-"):
                break
            universes.append(arg)
        if universes:
            allowed_universes = set(universes)

    run(ns, tf, cascade=cascade, allowed_universes=allowed_universes)
