from __future__ import annotations

# etf/guardrails.py

"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "ref"
"""
from core.paths import ROOT, REF #, CFG  # NEW

import pandas as pd
import numpy as np

from .trend_engine import load_etf_trend_scores



def aggregate_etf_score(row: pd.Series, cols: list[str]) -> float:
    """
    Combine primary / secondary ETF scores for one direction.

    Behavior:
      - If *both* columns are missing or NaN -> return np.nan (no ETF data)
      - Otherwise:
          - treat None / NaN as 0.0
          - return max(cleaned_scores)
    """
    raw_vals = []
    for c in cols:
        if c in row:
            raw_vals.append(row[c])
        else:
            raw_vals.append(np.nan)

    # If all values are missing/NaN, this symbol has no ETF mapping/data
    if all(pd.isna(v) for v in raw_vals):
        return np.nan

    cleaned = []
    for v in raw_vals:
        if v is None or pd.isna(v):
            cleaned.append(0.0)
        else:
            try:
                cleaned.append(float(v))
            except (TypeError, ValueError):
                cleaned.append(0.0)

    return max(cleaned)



def attach_etf_trends_for_options_combo(
    combo_df: pd.DataFrame,
    combo_cfg: dict,
    timeframe_for_etf: str = "weekly",
) -> pd.DataFrame:
    """
    For options-eligible combos, join ETF trend scores using the
    symbol_to_etf_options_eligible.csv mapping and the ETF weekly scores.

    Assumes mapping has at least:
        - symbol
        - etf_symbol_primary
    """
    universe = combo_cfg.get("universe")
    if universe != "options_eligible":
        # Nothing to do for shortlist / futures / anything else
        return combo_df

    mapping_path = REF / "symbol_to_etf_options_eligible.csv"
    if not mapping_path.exists():
        print(f"[WARN] ETF mapping not found at {mapping_path}; skipping ETF guardrail join.")
        return combo_df

    mapping = pd.read_csv(mapping_path)

    if "symbol" not in mapping.columns or "etf_symbol_primary" not in mapping.columns:
        print(
            "[WARN] symbol_to_etf_options_eligible.csv missing "
            "'symbol' and/or 'etf_symbol_primary'; skipping ETF join."
        )
        return combo_df

    #etf_scores = compute_etf_trend_scores(timeframe_for_etf)  # index=etf_symbol
    etf_scores = load_etf_trend_scores(timeframe_for_etf)
    etf_scores = etf_scores.reset_index()  # columns: etf_symbol, etf_long_score, etf_short_score

    # 1) attach primary ETF symbol to combo rows
    combo = combo_df.merge(
        mapping[["symbol", "etf_symbol_primary", "etf_symbol_secondary"]],
        on="symbol",
        how="left",
    )
  
    # 2) attach ETF scores for that primary ETF
    combo = combo.merge(
        etf_scores.rename(columns={"etf_symbol": "etf_symbol_primary"}),
        on="etf_symbol_primary",
        how="left",
    )
    # Rename scores to make their role clear
    combo = combo.rename(
        columns={
            "etf_long_score": "etf_primary_long_score",
            "etf_short_score": "etf_primary_short_score",
        }
    )

    # 3) attach ETF scores for the ** secondary ** ETF
    combo = combo.merge(
        etf_scores.rename(columns={"etf_symbol": "etf_symbol_secondary"}),
        on="etf_symbol_secondary",
        how="left",
    )
    # Rename scores to make their role clear
    combo = combo.rename(
        columns={
            "etf_long_score": "etf_secondary_long_score",
            "etf_short_score": "etf_secondary_short_score",
        }
    )

    return combo
