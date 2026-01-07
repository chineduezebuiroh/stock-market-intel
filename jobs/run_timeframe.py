from __future__ import annotations

# jobs/run_timeframe.py

import sys
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
    load_quarterly_from_monthly,
    load_yearly_from_monthly,
    load_futures_intraday,
    safe_load_eod,
)
from etl.window import parquet_path, update_fixed_window
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
print("[DEBUG] load_eod signature:", inspect.signature(load_eod))


DEV_MAX_STOCK_SYMBOLS_PER_TF = None  # set to None to disable the cap
EXCLUSIONS_FILE = CFG / "excluded_symbols.csv"


# Load timeframe config (structure only)
with open(CFG / "timeframes.yaml", "r") as f:
    TF_CFG = yaml.safe_load(f)

# Load multi-timeframe combos (for universes per timeframe)
with open(CFG / "multi_timeframe_combos.yaml", "r") as f:
    MTF_CFG = yaml.safe_load(f)

CASCADE = {
    "stocks": {
        "intraday_130m": ["daily", "weekly"],
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
    total = len(symbols)
    print(f"[INGEST] {namespace}:{timeframe} starting ingest for {total} symbols", flush=True)

    

    for idx, sym in enumerate(symbols, start=1):
        start_sym = time.perf_counter()
        print(f"[INGEST] {namespace}:{timeframe} [{idx}/{total}] {sym} ...")

        try:
            if namespace == "stocks" and timeframe == "intraday_130m":
                df_new = load_130m_from_5m(sym, session=session)
                
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
            continue

        # 🔹 NEW: explicitly log empty df_new
        if df_new is None or df_new.empty:
            elapsed = time.perf_counter() - start_sym
            print(f"[INGEST][SKIP] {namespace}:{timeframe} {sym} df_new empty/None after {elapsed:.1f}s", flush=True)
            continue
        
        parquet = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        
        if storage.exists(parquet):
            existing = storage.load_parquet(parquet)
            # 🔹 Drop legacy tuple-ish columns left over from old experiments
            bad_cols = [
                c for c in existing.columns
                if isinstance(c, str) and c.startswith("(")
            ]
            if bad_cols:
                existing = existing.drop(columns=bad_cols)
        else:
            existing = pd.DataFrame()

        merged = update_fixed_window(df_new, existing, window_bars)

        # --- futures intraday sanitization: drop invalid bars ---
        if namespace == "futures" and timeframe in ("intraday_1h", "intraday_4h"):
            ohlc = ["open", "high", "low", "close"]
            
            # Drop rows where all OHLC are NaN (your exact poison signature)
            #merged = merged[~merged[ohlc].isna().all(axis=1)]
            
            # Drop rows where ANY OHLC are NaN (your exact poison signature)
            #merged = merged.dropna(subset=["open","high","low","close"], how="any")
            merged = merged.dropna(subset=ohlc, how="any")


        # 🔹 NEW: log if merged goes empty
        if merged is None or merged.empty:
            elapsed = time.perf_counter() - start_sym
            print(f"[INGEST][SKIP] {namespace}:{timeframe} {sym} merged empty after {elapsed:.1f}s", flush=True)
            continue

        #merged = apply_core(merged, params={})
        merged = apply_core(merged, namespace=namespace, timeframe=timeframe)

        # 🔹 NEW: log if indicators somehow wipe it out
        if merged is None or merged.empty:
            elapsed = time.perf_counter() - start_sym
            print(f"[INGEST][SKIP] {namespace}:{timeframe} {sym} post-indicators empty after {elapsed:.1f}s", flush=True)
            continue

        storage.save_parquet(merged, parquet)
        
        # --- verify write landed where we think it did ---
        try:
            ok = storage.exists(parquet)
        except Exception as e:
            ok = False
            print(f"[WRITE][ERR] exists-check failed for {parquet}: {e}", flush=True)
        
        if not ok:
            print(f"[WRITE][MISS] {namespace}:{timeframe} {sym} -> {parquet}", flush=True)
        else:
            # optional: only print occasionally to avoid spam
            if idx <= 3 or idx % 50 == 0:
                print(f"[WRITE][OK] {namespace}:{timeframe} {sym} -> {parquet}", flush=True)


        if idx == 1:  # only for the first symbol each timeframe
            try:
                df_chk = storage.load_parquet(parquet)
                print(f"[WRITE][CHK] {namespace}:{timeframe} {sym} saved shape={df_chk.shape}", flush=True)
            except Exception as e:
                print(f"[WRITE][CHK][ERR] {namespace}:{timeframe} {sym}: {e}", flush=True)


        elapsed = time.perf_counter() - start_sym
        print(f"[INGEST] {namespace}:{timeframe} [{idx}/{total}] {sym} OK in {elapsed:.1f}s", flush=True)


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
    ingest_one(namespace, timeframe, symbols, session, window_bars)

    # 3) Build single-timeframe snapshot (no screening/pivoting for now)
    base_cols = get_snapshot_base_cols(namespace, timeframe)
    rows = []

    for sym in symbols:
        p = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        """if not p.exists():"""
        if not storage.exists(p):
            continue

        """df = pd.read_parquet(p)"""
        df = storage.load_parquet(p)
        if df.empty:
            continue

        # Take the last bar as a Series
        #last = df.iloc[-1]

        # Take the last bar as a Series (but futures intraday must be valid OHLC)
        if namespace == "futures" and timeframe in ("intraday_1h", "intraday_4h"):
            df_valid = df.dropna(subset=["open","high","low","close"], how="any")
            if df_valid.empty:
                continue
            last = df_valid.iloc[-1]
        else:
            last = df.iloc[-1]

        # Ensure all base columns are present
        missing = [c for c in base_cols if c not in last.index]
        if missing:
            raise KeyError(
                f"Snapshot missing columns {missing} for {namespace}/{timeframe}/{sym}"
            )

        row = last[base_cols].copy()
        row["symbol"] = sym
        rows.append(row)

    if rows:
        snap = pd.DataFrame(rows)

        # Enforce column order: base cols + symbol
        snap = snap[base_cols + ["symbol"]]

        # Optional: nice index name
        if snap.index.name is None:
            snap.index.name = "date"

        # Ensure all columns are plain strings
        snap.columns = snap.columns.astype(str)

        out = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
        """snap.to_parquet(out)"""
        storage.save_parquet(snap, out)
        print(f"[OK] Wrote snapshot: {out}")

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
