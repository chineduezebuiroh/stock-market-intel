import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from etl.sources import load_eod, load_130m_from_5m
from etl.window import parquet_path, update_fixed_window
from indicators.core import apply_core
from screens.engine import run_screen

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



def symbols_for_universe(universe: str) -> list[str]:
    """
    Map a universe name to a list of symbols from CSVs.
    """
    if universe == "shortlist_stocks":
        p = CFG / "shortlist_stocks.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    if universe == "shortlist_futures":
        p = CFG / "shortlist_futures.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    if universe == "options_eligible":
        p = ROOT / "ref" / "options_eligible.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    # Future: ETF universes, etc.
    return []


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
    all_syms: set[str] = set()
    for u in universes:
        for sym in symbols_for_universe(u):
            all_syms.add(sym)

    symbols = sorted(all_syms)

    # Dev-only throttle for stocks; futures usually small anyway
    if namespace == "stocks" and DEV_MAX_STOCK_SYMBOLS_PER_TF is not None:
        symbols = symbols[:DEV_MAX_STOCK_SYMBOLS_PER_TF]

    return symbols

"""
def ingest_one(namespace: str, timeframe: str, symbols, session: str, window_bars: int):
    
    #Ingest bars for a single namespace+timeframe over a list of symbols,
    #update fixed rolling window, apply indicators, and persist parquet.
    
    for sym in symbols:
        if namespace == "stocks" and timeframe == "intraday_130m":
            df_new = load_130m_from_5m(sym, session=session)
        else:
            df_new = load_eod(sym, start="2000-01-01", end=None, session=session)

        parquet = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_parquet(parquet) if parquet.exists() else pd.DataFrame()
        merged = update_fixed_window(df_new, existing, window_bars)
        merged = apply_core(merged, params={})
        merged.to_parquet(parquet)
"""

def ingest_one(namespace: str, timeframe: str, symbols, session: str, window_bars: int):
    """
    Ingest bars for a single namespace+timeframe over a list of symbols,
    update fixed rolling window, apply indicators, and persist parquet.

    This version:
      - uses load_eod() with a start date derived from timeframe+window_bars,
      - relies on update_fixed_window() to enforce the sliding window.
    """
    for sym in symbols:
        if namespace == "stocks" and timeframe == "intraday_130m":
            # intraday builder (you can also add timeout logic inside this)
            df_new = load_130m_from_5m(sym, session=session)
        else:
            df_new = load_eod(sym, timeframe=timeframe, window_bars=window_bars, session=session)

        if df_new is None or df_new.empty:
            # skip symbols that fail to load
            continue

        parquet = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        parquet.parent.mkdir(parents=True, exist_ok=True)

        if parquet.exists():
            existing = pd.read_parquet(parquet)
        else:
            existing = pd.DataFrame()

        merged = update_fixed_window(df_new, existing, window_bars)
        if merged is None or merged.empty:
            continue

        merged = apply_core(merged, params={})
        merged.to_parquet(parquet)




"""
def run(namespace: str, timeframe: str, cascade: bool = False):

    cfg_tf = TF_CFG[namespace][timeframe]
    session = cfg_tf["session"]
    window_bars = int(cfg_tf["window_bars"])

    # Determine symbols from all combos that use this timeframe
    symbols = symbols_for_timeframe(namespace, timeframe)
    if not symbols:
        print(f"[WARN] No symbols found for {namespace}:{timeframe} via combos.")
        return

    # 1) Ingest this timeframe
    ingest_one(namespace, timeframe, symbols, session, window_bars)

    # 2) Cascades (multi-timeframe ingest logic)
    if cascade:
        if namespace == "stocks":
            if timeframe == "intraday_130m":
                # 130m → daily + weekly
                for tf in ["daily", "weekly"]:
                    c = TF_CFG["stocks"][tf]
                    ingest_one("stocks", tf, symbols, c["session"], int(c["window_bars"]))

            elif timeframe == "daily":
                # daily → weekly + monthly
                for tf in ["weekly", "monthly"]:
                    c = TF_CFG["stocks"][tf]
                    ingest_one("stocks", tf, symbols, c["session"], int(c["window_bars"]))

            elif timeframe == "weekly":
                # weekly → monthly + quarterly
                for tf in ["monthly", "quarterly"]:
                    c = TF_CFG["stocks"][tf]
                    ingest_one("stocks", tf, symbols, c["session"], int(c["window_bars"]))

            elif timeframe == "monthly":
                # monthly → quarterly + yearly
                for tf in ["quarterly", "yearly"]:
                    c = TF_CFG["stocks"][tf]
                    ingest_one("stocks", tf, symbols, c["session"], int(c["window_bars"]))        

        elif namespace == "futures":
            if timeframe == "hourly":
                # 1h → 4h + daily
                for tf in ["four_hour", "daily"]:
                    c = TF_CFG["futures"][tf]
                    ingest_one("futures", tf, symbols, c["session"], int(c["window_bars"]))

            elif timeframe == "four_hour":
                # 4h → daily + weekly
                for tf in ["daily", "weekly"]:
                    c = TF_CFG["futures"][tf]
                    ingest_one("futures", tf, symbols, c["session"], int(c["window_bars"]))

            elif timeframe == "daily":
                # daily → weekly + monthly
                for tf in ["weekly", "monthly"]:
                    c = TF_CFG["futures"][tf]
                    ingest_one("futures", tf, symbols, c["session"], int(c["window_bars"]))

    # 3) Run single-timeframe screen if a config exists
    screen_path = CFG / "screens" / f"{timeframe}.yaml"
    if screen_path.exists():
        with open(screen_path, "r") as f:
            screen_cfg = yaml.safe_load(f)

        rows = []
        for sym in symbols:
            p = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            if df.empty:
                continue
            last = df.iloc[-1:].copy()
            last["symbol"] = sym
            rows.append(last)

        if rows:
            snap = pd.concat(rows)
            snap = run_screen(snap, screen_cfg)
            out = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
            snap.to_parquet(out)
"""

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

    # --- 2) Ingest this timeframe (per-symbol parquet with indicators) ---
    ingest_one(namespace, timeframe, symbols, session, window_bars)

    # --- 3) Build snapshot for this timeframe ---
    # Read latest bar for each symbol from its parquet
    rows = []
    for sym in symbols:
        p = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        last = df.iloc[-1:].copy()
        last["symbol"] = sym
        rows.append(last)

    if not rows:
        print(f"[WARN] No latest bars found for snapshot {namespace}:{timeframe}.")
        snap = pd.DataFrame()
    else:
        snap = pd.concat(rows, axis=0)

    # Optionally apply screen if config exists
    screen_path = CFG / "screens" / f"{timeframe}.yaml"
    if screen_path.exists() and not snap.empty:
        with open(screen_path, "r") as f:
            screen_cfg = yaml.safe_load(f)
        snap = run_screen(snap, screen_cfg)

    # Write snapshot regardless of screen presence
    out = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(out)
    print(f"[OK] Wrote snapshot: {out}")

    # --- 4) Cascade to higher timeframes (if requested) ---
    if cascade:
        child_tfs = CASCADE.get(namespace, {}).get(timeframe, [])
        for child_tf in child_tfs:
            # Important: child runs with cascade=False to avoid infinite recursion
            run(namespace, child_tf, cascade=False)



if __name__ == "__main__":
    ns = sys.argv[1]
    tf = sys.argv[2]
    cascade = "--cascade" in sys.argv
    run(ns, tf, cascade=cascade)
