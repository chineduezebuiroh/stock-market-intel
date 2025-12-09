from __future__ import annotations

# etf/universe.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"
REF = ROOT / "ref"
"""
from core.paths import ROOT, REF, CFG  # NEW

import pandas as pd


def load_etf_universe() -> pd.DataFrame:
    """
    Returns the ETF universe used for mapping, from
    config/shortlist_sector_etfs.csv
    """
    path = CFG / "shortlist_sector_etfs.csv"
    return pd.read_csv(path)


def load_options_universe() -> pd.DataFrame:
    """
    Returns the options-eligible universe, from ref/options_eligible.csv
    """
    path = REF / "options_eligible.csv"
    return pd.read_csv(path)
