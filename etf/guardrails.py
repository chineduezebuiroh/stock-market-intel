from __future__ import annotations

# etf/guardrails.py

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
    lower_tf: str | None = None,
    middle_tf: str = "weekly",
) -> pd.DataFrame:
    """
    For options-eligible combos, join ETF trend scores using
    symbol_to_etf_options_eligible.csv and ETF snapshots
    for BOTH lower and middle timeframes.

    Produces columns like:
      - etf_lower_primary_long_score / etf_lower_primary_short_score
      - etf_lower_secondary_long_score / etf_lower_secondary_short_score
      - etf_middle_primary_long_score / etf_middle_primary_short_score
      - etf_middle_secondary_long_score / etf_middle_secondary_short_score

    For backward-compatibility, we alias middle_* -> etf_primary_*/etf_secondary_*.
    """
    universe = combo_cfg.get("universe")
    if universe != "options_eligible":
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

    combo = combo_df.merge(
        mapping[["symbol", "etf_symbol_primary", "etf_symbol_secondary"]],
        on="symbol",
        how="left",
    )

    # ----------------------
    # Middle-timeframe ETF scores
    # ----------------------
    etf_middle = load_etf_trend_scores(middle_tf)  # index=etf_symbol
    etf_middle = etf_middle.reset_index()  # etf_symbol, etf_long_score, etf_short_score

    combo = combo.merge(
        etf_middle.rename(columns={"etf_symbol": "etf_symbol_primary"}),
        on="etf_symbol_primary",
        how="left",
    ).rename(
        columns={
            "etf_long_score": "etf_middle_primary_long_score",
            "etf_short_score": "etf_middle_primary_short_score",
        }
    )

    combo = combo.merge(
        etf_middle.rename(columns={"etf_symbol": "etf_symbol_secondary"}),
        on="etf_symbol_secondary",
        how="left",
    ).rename(
        columns={
            "etf_long_score": "etf_middle_secondary_long_score",
            "etf_short_score": "etf_middle_secondary_short_score",
        }
    )

    # ----------------------
    # Lower-timeframe ETF scores (optional)
    # ----------------------
    if lower_tf:
        etf_lower = load_etf_trend_scores(lower_tf)
        etf_lower = etf_lower.reset_index()

        combo = combo.merge(
            etf_lower.rename(columns={"etf_symbol": "etf_symbol_primary"}),
            on="etf_symbol_primary",
            how="left",
        ).rename(
            columns={
                "etf_long_score": "etf_lower_primary_long_score",
                "etf_short_score": "etf_lower_primary_short_score",
            }
        )

        combo = combo.merge(
            etf_lower.rename(columns={"etf_symbol": "etf_symbol_secondary"}),
            on="etf_symbol_secondary",
            how="left",
        ).rename(
            columns={
                "etf_long_score": "etf_lower_secondary_long_score",
                "etf_short_score": "etf_lower_secondary_short_score",
            }
        )

    # ----------------------
    # Backward-compatible aliases for middle timeframe
    # ----------------------
    for src, dst in [
        ("etf_middle_primary_long_score", "etf_primary_long_score"),
        ("etf_middle_primary_short_score", "etf_primary_short_score"),
        ("etf_middle_secondary_long_score", "etf_secondary_long_score"),
        ("etf_middle_secondary_short_score", "etf_secondary_short_score"),
    ]:
        if src in combo.columns and dst not in combo.columns:
            combo[dst] = combo[src]

    return combo
