import sys
from pathlib import Path

# Ensure project root is on sys.path
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


with open(CFG / "timeframes.yaml", "r") as f:
    TF_CFG = yaml.safe_load(f)


def symbols_for(universe: str):
    """
    Map a universe name to a list of symbols, using CSV files.
    """
    if universe == "options_eligible":
        p = ROOT / "ref" / "options_eligible.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    if universe == "shortlist_stocks":
        p = CFG / "shortlist_stocks.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    if universe == "shortlist_futures":
        p = CFG / "shortlist_futures.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []
    
    if universe == "shortlist_sector_etfs":
        p = CFG / "shortlist_sector_etfs.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    # fallback
    return []



def ingest_one(namespace: str, timeframe: str, symbols, session: str, window_bars: int):
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


def run(namespace: str, timeframe: str, cascade: bool = False):
    cfg = TF_CFG[namespace][timeframe]
    session = cfg["session"]
    window_bars = int(cfg["window_bars"])
    symbols = symbols_for(cfg["universe"])

    # 1) Ingest the primary timeframe
    ingest_one(namespace, timeframe, symbols, session, window_bars)

    # 2) Cascades for multi-timeframe alignment
    if cascade:
        if namespace == "stocks":
            if timeframe == "intraday_130m":
                # Refresh daily + weekly for same shortlist
                for tf in ["daily", "weekly"]:
                    c = TF_CFG["stocks"][tf]
                    ingest_one("stocks", tf, symbols, c["session"], int(c["window_bars"]))
            elif timeframe == "daily":
                # Refresh weekly + monthly for same universe
                for tf in ["weekly", "monthly"]:
                    c = TF_CFG["stocks"][tf]
                    ingest_one("stocks", tf, symbols, c["session"], int(c["window_bars"]))
        elif namespace == "futures":
            if timeframe == "hourly":
                # Hourly → Daily
                c = TF_CFG["futures"]["daily"]
                ingest_one("futures", "daily", symbols, c["session"], int(c["window_bars"]))
            elif timeframe == "four_hour":
                # 4H → Weekly
                c = TF_CFG["futures"]["weekly"]
                ingest_one("futures", "weekly", symbols, c["session"], int(c["window_bars"]))
            elif timeframe == "daily":
                # Daily → Monthly
                c = TF_CFG["futures"]["monthly"]
                ingest_one("futures", "monthly", symbols, c["session"], int(c["window_bars"]))

    # 3) Run screen if there is a YAML for this timeframe
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
            out = ROOT / "data" / f"snapshot_{namespace}_{timeframe}.parquet"
            snap.to_parquet(out)


if __name__ == "__main__":
    ns = sys.argv[1]
    tf = sys.argv[2]
    cascade = "--cascade" in sys.argv
    run(ns, tf, cascade=cascade)
