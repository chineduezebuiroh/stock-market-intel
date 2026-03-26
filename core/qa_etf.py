from __future__ import annotations
# core/qa_etf.py

import pandas as pd

from core.notify import notify_telegram_message


def send_etf_mapping_qa_sample(
    df: pd.DataFrame,
    *,
    sample_size: int = 50,
    seed: int = 42,
    fuzzy_only: bool = True,
) -> None:
    """
    Sample ETF mapping rows and send via Telegram for QA review.

    If fuzzy_only=True, only sample rows where etf_match_source == 'fuzzy'.
    """
    if df is None or df.empty:
        print("[QA] ETF mapping empty; skipping QA sample.")
        return

    work = df.copy()

    if fuzzy_only:
        if "etf_match_source" not in work.columns:
            print("[QA] etf_match_source missing; cannot apply fuzzy_only filter. Skipping.")
            return

        work = work[work["etf_match_source"].astype(str).str.lower() == "fuzzy"]

        if work.empty:
            print("[QA] No fuzzy rows available for ETF QA sample; skipping.")
            return

    n = min(sample_size, len(work))
    sample = work.sample(n=n, random_state=seed).copy()

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

    lines = []
    title = "📊 ETF QA SAMPLE (fuzzy-only)" if fuzzy_only else "📊 ETF QA SAMPLE"
    lines.append(title)
    lines.append(f"rows: {len(sample)}")
    lines.append("")

    for _, row in sample.iterrows():
        symbol = row.get("symbol", "")
        primary = row.get("etf_symbol_primary", "")
        secondary = row.get("etf_symbol_secondary", "")
        source = row.get("etf_match_source", "")
        ind_score = row.get("industry_score", "")
        sec_score = row.get("sector_score", "")

        try:
            ind_score_fmt = f"{float(ind_score):.2f}" if pd.notna(ind_score) else ""
        except Exception:
            ind_score_fmt = str(ind_score)

        try:
            sec_score_fmt = f"{float(sec_score):.2f}" if pd.notna(sec_score) else ""
        except Exception:
            sec_score_fmt = str(sec_score)

        line = (
            f"{symbol} | "
            f"P:{primary} "
            f"S:{secondary} | "
            f"src:{source} | "
            f"ind:{ind_score_fmt} "
            f"sec:{sec_score_fmt}"
        )
        lines.append(line)

    msg = "\n".join(lines)
    notify_telegram_message(msg)
