from __future__ import annotations

# etl/etf_matching.py

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from pathlib import Path

# Project roots
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"
REF = ROOT / "ref"

# -----------------------------------------------------------
# Tokenization + similarity
# -----------------------------------------------------------

_STOPWORDS = {
    "etf",
    "fund",
    "sector",
    "select",
    "spdr",
    "ishares",
    "index",
    "trust",
    "inc",
    "corp",
    "corporation",
    "the",
    "and",
    "&",
    "usd",
}



def _tokenize(text: str | None) -> set[str]:
    """Very simple tokenizer: lowercase, strip punctuation, drop stopwords."""
    if not isinstance(text, str) or not text:
        return set()

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)  # non-alphanum -> space
    tokens = [t for t in text.split() if t and t not in _STOPWORDS]
    return set(tokens)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# -----------------------------------------------------------
# DTOs
# -----------------------------------------------------------

@dataclass
class EtfInfo:
    symbol: str
    name: str
    tokens: set[str]


# -----------------------------------------------------------
# Public API
# -----------------------------------------------------------

def build_symbol_to_etf_map(
    stocks_meta: pd.DataFrame,
    etfs_df: pd.DataFrame,
    *,
    industry_min_score: float = 0.20,
    sector_min_score: float = 0.10,
    default_etf: str = "SPY",
) -> pd.DataFrame:
    """
    Build a stock -> ETF mapping based purely on token similarity.

    stocks_meta: must have columns:
        - 'symbol'
        - 'sector'
        - 'industry'
      (If you also have 'long_name', you can optionally concatenate it into
       the industry text if you want more richness.)

    etfs_df: must have columns:
        - 'symbol'
        - 'name'   (ETF name / description from shortlist_sector_etfs.csv)

    Logic per stock:
      1) Tokenize industry & sector separately.
      2) For each ETF:
           - industry_score = jaccard(tokens(industry), tokens(etf_name))
           - sector_score   = jaccard(tokens(sector),   tokens(etf_name))
      3) If best industry_score >= industry_min_score -> choose its ETF.
         Else, if best sector_score >= sector_min_score -> choose its ETF.
         Else, choose default_etf.

    Returns DataFrame with columns:
        ['symbol', 'etf_symbol', 'industry_score', 'sector_score', 'chosen_by']
    """
    required_stock_cols = {"symbol", "sector", "industry"}
    missing_stock = required_stock_cols - set(stocks_meta.columns)
    if missing_stock:
        raise KeyError(f"stocks_meta missing columns: {sorted(missing_stock)}")

    required_etf_cols = {"symbol", "name"}
    missing_etf = required_etf_cols - set(etfs_df.columns)
    if missing_etf:
        raise KeyError(f"etfs_df missing columns: {sorted(missing_etf)}")

    # Precompute ETF tokens
    etfs: list[EtfInfo] = []
    for _, row in etfs_df.iterrows():
        etf_sym = str(row["symbol"])
        etf_name = str(row["name"])
        etfs.append(
            EtfInfo(
                symbol=etf_sym,
                name=etf_name,
                tokens=_tokenize(etf_name),
            )
        )

    rows: list[dict] = []

    for _, s in stocks_meta.iterrows():
        sym = str(s["symbol"])
        sector = str(s["sector"])
        industry = str(s["industry"])

        industry_tokens = _tokenize(industry)
        sector_tokens = _tokenize(sector)

        best_industry_score = 0.0
        best_industry_etf = None

        best_sector_score = 0.0
        best_sector_etf = None

        for etf in etfs:
            # industry-based similarity
            ind_score = _jaccard(industry_tokens, etf.tokens)
            if ind_score > best_industry_score:
                best_industry_score = ind_score
                best_industry_etf = etf.symbol

            # sector-based similarity
            sec_score = _jaccard(sector_tokens, etf.tokens)
            if sec_score > best_sector_score:
                best_sector_score = sec_score
                best_sector_etf = etf.symbol

        # Choose ETF according to your rule:
        #  - prefer strong industry match
        #  - else fall back to sector
        #  - else default
        if best_industry_etf is not None and best_industry_score >= industry_min_score:
            chosen_etf = best_industry_etf
            chosen_by = "industry"
        elif best_sector_etf is not None and best_sector_score >= sector_min_score:
            chosen_etf = best_sector_etf
            chosen_by = "sector"
        else:
            chosen_etf = default_etf
            chosen_by = "default"

        rows.append(
            {
                "symbol": sym,
                "sector": sector,
                "industry": industry,
                "etf_symbol": chosen_etf,
                "industry_score": best_industry_score,
                "industry_etf": best_industry_etf,
                "sector_score": best_sector_score,
                "sector_etf": best_sector_etf,
                "chosen_by": chosen_by,
            }
        )

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Public entrypoint: build ETF mapping for options-eligible only
# ---------------------------------------------------------------------

def build_options_eligible_etf_map() -> pd.DataFrame:
    """
    Build a symbol -> ETF mapping for *options-eligible* stocks.

    In this setup, the security_master file *is* your options-eligible universe:
      - ref/security_master.parquet  (must have: symbol, sector, industry)

    ETF universe:
      - config/shortlist_sector_etfs.csv  (symbol, name, ...)

    Output:
      - ref/symbol_to_etf_options_eligible.csv
    """
    sec_master_path = REF / "options_eligible.csv"
    etf_path = CFG / "shortlist_sector_etfs.csv"

    if not sec_master_path.exists():
        raise FileNotFoundError(f"Missing security master: {sec_master_path}")

    if not etf_path.exists():
        raise FileNotFoundError(f"Missing ETF shortlist: {etf_path}")

    # 1) Load security master (this *is* your options-eligible universe)
    #sec_master = pd.read_parquet(sec_master_path)
    sec_master = pd.read_csv(sec_master_path)

    required_cols = {"symbol", "sector", "industry"}
    missing = required_cols - set(sec_master.columns)
    if missing:
        raise KeyError(
            f"security_master.parquet missing columns: {sorted(missing)} "
            f"(have: {sorted(sec_master.columns)})"
        )

    # This already *is* options-eligible; just drop sector/industry nulls
    stocks_meta = sec_master.dropna(subset=["sector", "industry"]).copy()

    # 2) Load ETF universe
    etfs_df = pd.read_csv(etf_path)
    if "symbol" not in etfs_df.columns or "name" not in etfs_df.columns:
        raise KeyError(
            "shortlist_sector_etfs.csv must have 'symbol' and 'name' columns"
        )

    # 3) Build mapping using your tokenized similarity helper
    mapping = build_symbol_to_etf_map(
        stocks_meta=stocks_meta,
        etfs_df=etfs_df,
        industry_min_score=0.35,  # tweakable
        sector_min_score=0.25,    # tweakable
    )

    out = REF / "symbol_to_etf_options_eligible.csv"
    mapping.to_csv(out, index=False)
    print(
        f"[OK] Wrote ETF mapping for options-eligible symbols "
        f"({len(mapping)} rows) to {out}"
    )

    return mapping


if __name__ == "__main__":
    build_options_eligible_etf_map()
