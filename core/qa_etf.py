from __future__ import annotations
# core/qa_etf.py

import pandas as pd
import numpy as np

from core.notify import notify_combo_signals  # reuse your Telegram pipe


def send_etf_mapping_qa_sample(
    df: pd.DataFrame,
    *,
    sample_size: int = 50,
    seed: int = 42,
) -> None:
    """
    Sample ETF mapping rows and send via Telegram for QA review.
    """

    if df is None or df.empty:
        print("[QA] ETF mapping empty; skipping QA sample.")
        return

    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=seed).copy()

    cols = [
        "symbol",
        "sector",
        "industry",
        "industry_etf",
        "industry_score",
        "sector_etf",
        "sector_score",
        "etf_symbol_primary",
        "etf_symbol_secondary",
        "etf_match_source",
        "etf_override_notes",
    ]

    existing = [c for c in cols if c in sample.columns]
    sample = sample[existing]

    # Format as text block (Telegram-friendly)
    lines = ["📊 ETF QA SAMPLE (random 50):\n"]

    for _, row in sample.iterrows():
        line = (
            f"{row.get('symbol')} | "
            f"P:{row.get('etf_symbol_primary')} "
            f"S:{row.get('etf_symbol_secondary')} | "
            f"src:{row.get('etf_match_source')} | "
            f"ind:{round(row.get('industry_score', 0), 2)} "
            f"sec:{round(row.get('sector_score', 0), 2)}"
        )
        lines.append(line)

    msg = "\n".join(lines)

    notify_combo_signals(msg)
  
