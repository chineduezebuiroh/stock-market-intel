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
from core.paths import REF, CFG  # NEW

# -----------------------------------------------------------
# Tokenization + similarity
# -----------------------------------------------------------

_token_re = re.compile(r"[A-Za-z0-9]+")

_STOPWORDS = {
    "500",
    "&",
    "and",
    "care",
    #"consumer",
    "corp",
    "corporation",
    "diversified",
    "equal",
    "etf",
    "exchange",
    "expanded",
    "fund",
    "global",
    "inc",
    "index",
    "invesco",
    "ishares",
    "kbw"
    "regional",
    "s&p",
    "sector",
    "select",
    "services",
    "spdr",
    "the",
    "trust",
    "usd",    
    "vaneck",
    "weight",
}

# A few known exceptions where trailing "s" is *not* just a plural.
_PLURAL_EXCEPTIONS = {
    "ms", "us", "gas", "as", "is", "this", "yes"
}

# Domain-specific “stem” map for tricky tokens
_SPECIAL_STEMS = {
    "healthcare": "health",
    "health": "health",
    "jets": "airlines",
    "airlines": "airlines",
    "discretionary": "cyclical",
    "cyclical": "cyclical",
    "staples": "defensive",
    "defensive": "defensive",
    # you can extend this later, e.g.:
    # "utilities": "utility", etc., but plural handling already helps a lot
}

def _normalize_token(tok: str) -> str:
    """
    Domain-aware normalization:
      - lowercase
      - map special stems (healthcare -> health)
      - 'ies' -> 'y' (technologies -> technology)
      - strip trailing 's' for simple plurals (industrials -> industrial)
    """
    tok = tok.lower()

    # special domain stems first
    if tok in _SPECIAL_STEMS:
        return _SPECIAL_STEMS[tok]

    if len(tok) <= 3:
        return tok

    if tok in _PLURAL_EXCEPTIONS:
        return tok

    # technologies -> technology, utilities -> utility
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"

    # simple plural: industrials -> industrial, banks -> bank
    if tok.endswith("s"):
        return tok[:-1]

    return tok
    

def _tokenize(text: str | None) -> set[str]:
    """Tokenize with stopword removal + plural/inflection normalization."""
    if not isinstance(text, str) or not text:
        return set()

    raw_tokens = _token_re.findall(text.lower())

    tokens: set[str] = set()
    for t in raw_tokens:
        if not t or t in _STOPWORDS:
            continue
        norm = _normalize_token(t)
        if norm and norm not in _STOPWORDS:
            tokens.add(norm)

    return tokens


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
    industry_min_score: float = 0.20,
    sector_min_score: float = 0.50,
    default_etf: str = "",
) -> pd.DataFrame:
    """
    Build a stock -> ETF mapping using Jaccard over normalized tokens.

    stocks_meta: must have columns:
        - 'symbol'
        - 'sector'
        - 'industry'
        - (optional) 'name'  (company name; we’ll pass it through)

    etfs_df: must have columns:
        - 'symbol'
        - 'name'   (ETF name / description)

    For each stock:
      - Compute best industry ETF (industry_score, industry_etf, industry_etf_name)
      - Compute best sector ETF   (sector_score,   sector_etf,   sector_etf_name)

    Then derive:
      - etf_symbol_primary
      - etf_symbol_secondary

    Primary rule (tweakable later):
      1) If best industry_score >= industry_min_score -> primary = industry_etf
      2) Else if best sector_score >= sector_min_score -> primary = sector_etf
      3) Else -> primary = default_etf

    Secondary rule (also tweakable):
      - If the other ETF exists, has score > 0, and != primary -> secondary = that ETF
      - Else secondary = None
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
    etfs = []
    for _, row in etfs_df.iterrows():
        etf_sym = str(row["symbol"])
        etf_name = str(row["name"])
        etfs.append(
            {
                "symbol": etf_sym,
                "name": etf_name,
                "tokens": _tokenize(etf_name),
            }
        )

    rows: list[dict] = []

    for _, s in stocks_meta.iterrows():
        sym = str(s["symbol"])
        sector = str(s["sector"])
        industry = str(s["industry"])
        name = str(s.get("name", ""))  # stock’s long name if present

        industry_tokens = _tokenize(industry)
        sector_tokens = _tokenize(sector)

        best_industry_score = 0.0
        best_industry_etf = None
        best_industry_name = None

        best_sector_score = 0.0
        best_sector_etf = None
        best_sector_name = None

        for etf in etfs:
            etf_tokens = etf["tokens"]

            ind_score = _jaccard(industry_tokens, etf_tokens)
            if ind_score > best_industry_score:
                best_industry_score = ind_score
                best_industry_etf = etf["symbol"]
                best_industry_name = etf["name"]

            sec_score = _jaccard(sector_tokens, etf_tokens)
            if sec_score > best_sector_score:
                best_sector_score = sec_score
                best_sector_etf = etf["symbol"]
                best_sector_name = etf["name"]

        # --- Primary selection (you can tweak this logic later) ---
        if best_industry_etf is not None and best_industry_score >= industry_min_score:
            primary = best_industry_etf
        elif best_sector_etf is not None and best_sector_score >= sector_min_score:
            primary = best_sector_etf
        else:
            primary = default_etf

        # --- Secondary selection (keep the "other" candidate if meaningful) ---
        secondary = None
        # Candidate list: (etf_symbol, score)
        candidates = [
            (best_industry_etf, best_industry_score, industry_min_score),
            (best_sector_etf, best_sector_score, sector_min_score),
        ]
        for etf_sym, score, th in candidates:
            if etf_sym is None:
                continue
            if etf_sym == primary:
                continue
            if score < th: #<--- previously was score <= 0.0:
                continue
            secondary = etf_sym
            break  # first non-primary, sufficiently positive-score candidate

        rows.append(
            {
                "symbol": sym,
                "name": name,
                "sector": sector,
                "industry": industry,
                "industry_etf": best_industry_etf,
                "industry_etf_name": best_industry_name,
                "industry_score": float(best_industry_score),
                "sector_etf": best_sector_etf,
                "sector_etf_name": best_sector_name,
                "sector_score": float(best_sector_score),
                "etf_symbol_primary": primary,
                "etf_symbol_secondary": secondary,
            }
        )

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Public entrypoint: build ETF mapping for options-eligible only
# ---------------------------------------------------------------------



if __name__ == "__main__":
    build_options_eligible_etf_map()
