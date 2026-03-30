from __future__ import annotations
# core/qa_etf.py

import io
import pandas as pd

from core.notify import send_email_message


def send_etf_mapping_qa_sample(
    df: pd.DataFrame,
    *,
    sample_size: int = 50,
    seed: int = 42,
    fuzzy_only: bool = True,
) -> None:
    """
    Sample ETF mapping rows and send via email for QA review.

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
        "name",
        "sector",
        "industry",
        "industry_etf",
        "industry_etf_name",
        "industry_score",
        "sector_etf",
        "sector_etf_name",
        "sector_score",
        "etf_symbol_primary",
        "etf_symbol_secondary",
        "etf_match_source",
        "etf_override_notes",
    ]
    existing = [c for c in cols if c in sample.columns]
    sample = sample[existing]

    body_lines = [
        "ETF mapping QA sample attached.",
        "",
        f"Rows included: {len(sample)}",
        f"Filter: {'fuzzy-only' if fuzzy_only else 'all rows'}",
        "",
        "This is intended for manual review of ETF-to-symbol matching quality.",
    ]
    body = "\n".join(body_lines)

    csv_buf = io.StringIO()
    sample.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    send_email_message(
        subject="ETF Mapping QA Sample",
        body=body,
        attachments=[
            ("etf_mapping_qa_sample.csv", csv_bytes, "text/csv"),
        ],
    )

    print(f"[QA] Sent ETF QA sample email ({len(sample)} rows).")
    
