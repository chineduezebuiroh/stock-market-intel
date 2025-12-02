from __future__ import annotations

# jobs/build_symbol_to_etf_mapping.py

import sys
from pathlib import Path

import pandas as pd

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etl.etf_matching import build_symbol_to_etf_map  # type: ignore


DATA = ROOT / "data"
REF = ROOT / "ref"
CFG = ROOT / "config"


def main():
    # 1) Load stock metadata (wherever you keep sector/industry per symbol)
    # Adjust this to your actual security master.
    #
    # Example: ref/security_master.parquet with columns:
    #   symbol, sector, industry, ...
    sec_master = REF / "security_master.parquet"
    stocks_meta = pd.read_parquet(sec_master)

    # Optionally filter to universe(s) you care about:
    # e.g. only options_eligible
    # opts = pd.read_csv(REF / "options_eligible.csv")["symbol"].unique()
    # stocks_meta = stocks_meta[stocks_meta["symbol"].isin(opts)]

    # 2) Load ETF universe from your shortlist file
    etfs_df = pd.read_csv(CFG / "shortlist_sector_etfs.csv")

    # 3) Build mapping
    mapping = build_symbol_to_etf_map(stocks_meta, etfs_df)

    # 4) Persist mapping
    REF.mkdir(exist_ok=True, parents=True)
    out = REF / "symbol_to_etf.csv"
    mapping.to_csv(out, index=False)

    print(f"[OK] Wrote symbol->ETF mapping to {out}")
    print(mapping.head(10))


if __name__ == "__main__":
    main()
