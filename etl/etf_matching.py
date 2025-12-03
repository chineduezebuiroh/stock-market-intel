from __future__ import annotations

# etl/etf_matching.py

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from pathlib import Path

import difflib

# Project roots
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"
REF = ROOT / "ref"

_token_re = re.compile(r"[A-Za-z0-9]+")

# -----------------------------------------------------------
# Tokenization + similarity
# -----------------------------------------------------------

_STOPWORDS = {
    "etf",
    "fund",
    "sector",
    "select",
    "spdr",
    "kbw"
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
    "regional",
    "invesco",
    "diversified",
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


def _normalize_for_fuzzy(s: str) -> str:
    """
    Lowercase, tokenize to alphanumerics, drop stopwords, dedupe+sort tokens.

    This keeps your old stopword behavior while making the string friendlier
    for fuzzy matching.
    """
    if not isinstance(s, str):
        return ""
    tokens = _token_re.findall(s.lower())
    if _STOPWORDS:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    tokens = sorted(set(tokens))
    return " ".join(tokens)


def _fuzzy_similarity(a: str, b: str, stopwords: set[str] | None = None) -> float:
    """
    Return similarity in [0, 1] using SequenceMatcher on normalized strings.
    """
    na = _normalize_for_fuzzy(a)
    nb = _normalize_for_fuzzy(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()

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
    industry_min_score: float = 0.50,
    sector_min_score: float = 0.50,
    default_etf: str = "SPY",
) -> pd.DataFrame:
    """
    Build a stock -> ETF mapping based on fuzzy similarity between
    sector/industry strings and ETF names.

    stocks_meta: must have columns:
        - 'symbol'
        - 'sector'
        - 'industry'

    etfs_df: must have columns:
        - 'symbol'
        - 'name'   (ETF name / description from shortlist_sector_etfs.csv)

    Logic per stock:
      1) Compute fuzzy similarity between industry and each ETF name.
      2) Compute fuzzy similarity between sector and each ETF name.
      3) If best industry_score >= industry_min_score -> choose its ETF.
         Else, if best sector_score >= sector_min_score -> choose its ETF.
         Else, choose default_etf.

    Returns DataFrame with columns:
        [
          'symbol', 'sector', 'industry', 'etf_symbol',
          'industry_score', 'industry_etf', 'industry_etf_name',
          'sector_score', 'sector_etf', 'sector_etf_name',
          'chosen_by'
        ]
    """
    required_stock_cols = {"symbol", "sector", "industry"}
    missing_stock = required_stock_cols - set(stocks_meta.columns)
    if missing_stock:
        raise KeyError(f"stocks_meta missing columns: {sorted(missing_stock)}")

    required_etf_cols = {"symbol", "name"}
    missing_etf = required_etf_cols - set(etfs_df.columns)
    if missing_etf:
        raise KeyError(f"etfs_df missing columns: {sorted(missing_etf)}")

    # We'll just iterate directly over the ETF rows and call _fuzzy_similarity().
    rows: list[dict] = []

    for _, s in stocks_meta.iterrows():
        sym = str(s["symbol"])
        name = str(s["name"])  # safely pull company name if available
        sector = str(s["sector"])
        industry = str(s["industry"])

        best_industry_score = 0.0
        best_industry_etf = None
        best_industry_name = None

        best_sector_score = 0.0
        best_sector_etf = None
        best_sector_name = None

        for _, e in etfs_df.iterrows():
            etf_sym = str(e["symbol"])
            etf_name = str(e["name"])

            # Fuzzy similarity uses normalized (tokenized + stopword-stripped) strings
            ind_score = _fuzzy_similarity(industry, etf_name)
            if ind_score > best_industry_score:
                best_industry_score = ind_score
                best_industry_etf = etf_sym
                best_industry_name = etf_name

            sec_score = _fuzzy_similarity(sector, etf_name)
            if sec_score > best_sector_score:
                best_sector_score = sec_score
                best_sector_etf = etf_sym
                best_sector_name = etf_name

        # Choose ETF according to the hierarchy:
        #  - prefer strong industry match
        #  - else fall back to sector
        #  - else default
        """
        if best_industry_etf is not None and best_industry_score >= industry_min_score:
            chosen_etf = best_industry_etf
            chosen_by = "industry"
        elif best_sector_etf is not None and best_sector_score >= sector_min_score:
            chosen_etf = best_sector_etf
            chosen_by = "sector"
        else:
            chosen_etf = default_etf
            chosen_by = "default"
        """

        if best_industry_etf is not None and best_sector_etf is not None and best_industry_score > industry_min_score and best_sector_score > sector_min_score:
            if best_industry_score >= best_sector_score:
                chosen_etf1 = best_industry_etf
                chosen_etf2 = best_sector_etf
            else:
                chosen_etf1 = best_sector_etf
                chosen_etf2 = best_industry_etf
        elif (best_industry_etf is not None and best_sector_etf is None) or (best_industry_score > industry_min_score and best_sector_score <= sector_min_score):
            chosen_etf1 = best_industry_etf
            chosen_etf2 = ""
        elif (best_industry_etf is None and best_sector_etf is not None) or (best_industry_score <= industry_min_score and best_sector_score > sector_min_score):
            chosen_etf1 = best_sector_etf
            chosen_etf2 = ""
        else:
            chosen_etf1 = ""
            chosen_etf2 = ""

        rows.append(
            {
                "symbol": sym,
                "name": name,
                "sector": sector,
                "industry": industry,
                "etf_match_1": chosen_etf1,
                "etf_match_2": chosen_etf2,
                "industry_score": float(best_industry_score),
                "industry_etf": best_industry_etf,
                "industry_etf_name": best_industry_name,  # NEW
                "sector_score": float(best_sector_score),
                "sector_etf": best_sector_etf,
                "sector_etf_name": best_sector_name,      # NEW
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
