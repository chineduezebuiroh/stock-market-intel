from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"


def symbols_for_universe(universe: str) -> list[str]:
    """
    Map a universe name to a list of symbols from CSVs.

    Single source of truth used by both:
      - jobs/run_timeframe.py
      - jobs/run_combo.py
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
