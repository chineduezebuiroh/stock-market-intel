from __future__ import annotations

# etf/mapping_engine.py

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""
REF = ROOT / "ref"
"""
from core.paths import REF #, CFG  # NEW

import pandas as pd

from .universe import load_etf_universe, load_options_universe
from etl.etf_matching import build_symbol_to_etf_map  # reuse existing logic



def build_options_eligible_etf_map() -> pd.DataFrame:
    """
    Build a symbol -> ETF mapping for *options-eligible* stocks.

    In this setup, the security_master file *is* your options-eligible universe:
      - ref/security_master.parquet  (must have: symbol, sector, industry)

    ETF universe:
      - config/shortlist_sector_etfs.csv  (symbol, name, ...)
    """

    # 1) Load security master (this *is* your options-eligible universe)
    opts = load_options_universe()

    required_cols = {"symbol", "sector", "industry"}
    missing = required_cols - set(opts.columns)
    if missing:
        raise KeyError(
            f"security_master.parquet missing columns: {sorted(missing)} "
            f"(have: {sorted(opts.columns)})"
        )

    # This already *is* options-eligible; just drop sector/industry nulls
    stocks_meta = opts.dropna(subset=["sector", "industry"]).copy()

    # 2) Load ETF universe
    etfs_df = load_etf_universe()
    if "symbol" not in etfs_df.columns or "name" not in etfs_df.columns:
        raise KeyError(
            "shortlist_sector_etfs.csv must have 'symbol' and 'name' columns"
        )

    # 3) Build mapping using your tokenized similarity helper
    mapping = build_symbol_to_etf_map(
        stocks_meta=stocks_meta,
        etfs_df=etfs_df,
        #industry_min_score=0.35,  # tweakable
        #sector_min_score=0.25,    # tweakable
    )

    return mapping


def write_options_etf_mapping() -> Path:
    """
    Compute and persist mapping to:
        ref/symbol_to_etf_options_eligible.csv
    """
    mapping = build_options_eligible_etf_map()
    out = REF / "symbol_to_etf_options_eligible.csv"
    mapping.to_csv(out, index=False)
        print(
        f"[OK] Wrote ETF mapping for options-eligible symbols "
        f"({len(mapping)} rows) to {out}"
    )
    return out
