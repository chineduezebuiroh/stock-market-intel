from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from etl.sources import load_eod, load_130m_from_5m, load_yearly_from_monthly
from etl.window import parquet_path, update_fixed_window

from etl.universe import symbols_for_universe

#from indicators.core import apply_core
from indicators.core import (
    apply_core,
    get_snapshot_base_cols,
    initialize_indicator_engine,
)
    
DATA = ROOT / "data"
CFG = ROOT / "config"



DEV_MAX_STOCK_SYMBOLS_PER_TF = 50  # set to None to disable the cap

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
        "hourly": ["four_hour", "daily"],
        "four_hour": ["daily", "weekly"],
        "daily": ["weekly", "monthly"],
    },
}


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



def symbols_for_timeframe(namespace: str, timeframe: str) -> list[str]:
    """
    Union of all symbols for all universes that reference this timeframe,
    with a dev-only cap for stocks so local runs stay fast.
    """
    universes = universes_for_timeframe(namespace, timeframe)

    shortlist_syms: set[str] = set()
    other_syms: set[str] = set()

    for u in universes:
        syms = set(symbols_for_universe(u))
        if u.startswith("shortlist_"):
            shortlist_syms.update(syms)
        else:
            other_syms.update(syms)

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
    for sym in symbols:
        if namespace == "stocks" and timeframe == "intraday_130m":
            df_new = load_130m_from_5m(sym, session=session)
        elif namespace == "stocks" and timeframe == "yearly":
            df_new = load_yearly_from_monthly(sym, window_bars=window_bars, session=session)
        else:
            df_new = load_eod(sym, timeframe=timeframe, window_bars=window_bars, session=session)

        if df_new is None or df_new.empty:
            # skip symbols that fail to load
            continue

        parquet = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        parquet.parent.mkdir(parents=True, exist_ok=True)

        if parquet.exists():
            existing = pd.read_parquet(parquet)
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
        if merged is None or merged.empty:
            continue

        #merged = apply_core(merged, params={})
        merged = apply_core(merged, namespace=namespace, timeframe=timeframe)
        
        merged.to_parquet(parquet)



def run(namespace: str, timeframe: str, cascade: bool = False):
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

    symbols = symbols_for_timeframe(namespace, timeframe)
    if not symbols:
        print(f"[WARN] No symbols found for {namespace}:{timeframe} via combos.")
        return
    
    """
    # Initialize indicator engine (loads YAML profiles/params/combos)
    initialize_indicator_engine("config")
    """

    # --- 2) Ingest this timeframe (per-symbol parquet with indicators) ---
    ingest_one(namespace, timeframe, symbols, session, window_bars)


    # 3) Build single-timeframe snapshot (no screening/pivoting for now)
    base_cols = get_snapshot_base_cols(namespace, timeframe)
    rows = []

    for sym in symbols:
        p = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        if not p.exists():
            continue

        df = pd.read_parquet(p)
        if df.empty:
            continue

        # Take the last bar as a Series
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
        snap.to_parquet(out)
        print(f"[OK] Wrote snapshot: {out}")


    # --- 4) Cascade to higher timeframes (if requested) ---
    if cascade:
        child_tfs = CASCADE.get(namespace, {}).get(timeframe, [])
        for child_tf in child_tfs:
            # Important: child runs with cascade=False to avoid infinite recursion
            run(namespace, child_tf, cascade=False)



if __name__ == "__main__":
    initialize_indicator_engine(CFG)
    ns = sys.argv[1]
    tf = sys.argv[2]
    cascade = "--cascade" in sys.argv
    run(ns, tf, cascade=cascade)
