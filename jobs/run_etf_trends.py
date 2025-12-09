# jobs/run_etf_trends.py

import sys

"""
from pathlib import Path
# Project root + sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
CFG = ROOT / "config"
REF = ROOT / "ref"
"""
from core.paths import ROOT, DATA, CFG, REF  # NEW

import pandas as pd
import yaml

from etl.sources import load_eod
from etl.window import parquet_path, update_fixed_window
from indicators.core import (
    apply_core,
    initialize_indicator_engine,
)


def load_timeframe_cfg(namespace: str, timeframe: str) -> tuple[str, int]:
    """
    Reuse your timeframes.yaml for window_bars + session.
    """
    with open(CFG / "timeframes.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    tf_cfg = cfg[namespace][timeframe]
    session = tf_cfg["session"]
    window_bars = int(tf_cfg["window_bars"])
    return session, window_bars


def load_etf_symbols() -> list[str]:
    """
    ETF universe comes from config/shortlist_sector_etfs.csv (symbol, name, ...).
    """
    csv_path = CFG / "shortlist_sector_etfs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing ETF shortlist: {csv_path}")
    df = pd.read_csv(csv_path)
    if "symbol" not in df.columns:
        raise KeyError("shortlist_sector_etfs.csv must have a 'symbol' column")
    syms = sorted(df["symbol"].dropna().astype(str).unique())

    return syms


def ingest_etf_timeframe(timeframe: str) -> None:
    """
    Ingest ETFs for a single timeframe (e.g. weekly) and build a snapshot
    with indicators applied.
    """
    namespace = "stocks"  # reuse same indicator configs
    session, window_bars = load_timeframe_cfg(namespace, timeframe)

    initialize_indicator_engine(CFG)

    symbols = load_etf_symbols()
    if not symbols:
        print("[WARN] No ETF symbols found.")
        return

    rows = []
    for sym in symbols:
        df_new = load_eod(sym, timeframe=timeframe, window_bars=window_bars, session=session)
        if df_new is None or df_new.empty:
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

        merged = apply_core(merged, namespace=namespace, timeframe=timeframe)
        merged.to_parquet(parquet)

        last = merged.iloc[-1:].copy()
        last["symbol"] = sym
        rows.append(last)

    if not rows:
        print(f"[WARN] No ETF rows ingested for timeframe={timeframe}")
        return

    snap = pd.concat(rows, axis=0)
    if snap.index.name is None:
        snap.index.name = "date"
    snap.columns = snap.columns.astype(str)

    out = DATA / f"snapshot_etf_{timeframe}.parquet"
    snap.to_parquet(out)
    print(f"[OK] Wrote ETF snapshot: {out}")


if __name__ == "__main__":
    # default to weekly if not provided
    tf = sys.argv[1] if len(sys.argv) > 1 else "weekly"
    ingest_etf_timeframe(tf)
