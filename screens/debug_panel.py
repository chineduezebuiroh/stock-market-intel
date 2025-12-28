from __future__ import annotations

# screens/debug_panel.py

import numpy as np
import pandas as pd
import streamlit as st

from app.field_catalog import DEBUG_FIELDS


def _fmt_val(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return f"{v:.2f}"
        return str(v)
    except Exception:
        return str(v)


def _render_tf_block(col_container, row: pd.Series, prefix: str, title: str):
    col_container.markdown(f"### {title}")

    lines = []
    for fs in DEBUG_FIELDS:
        col_name = f"{prefix}_{fs.suffix}"
        if col_name in row.index:
            lines.append(f"- **{fs.label}**: {_fmt_val(row[col_name])}")

    if lines:
        col_container.markdown("\n".join(lines))
    else:
        col_container.info("No fields available for this timeframe.")


def render_symbol_debug_panel(
    df_full: pd.DataFrame,
    *,
    label: str = "Inspect symbol",
    key: str | None = None,
):
    if df_full is None or df_full.empty or "symbol" not in df_full.columns:
        return

    symbols = sorted(df_full["symbol"].unique())
    if not symbols:
        return

    with st.expander(label, expanded=False):
        sym = st.selectbox(
            "Symbol",
            options=symbols,
            key=f"{key}_symbol_select" if key else None,
        )

        df_sym = df_full[df_full["symbol"] == sym]
        if df_sym.empty:
            st.info("No rows for selected symbol.")
            return

        row = df_sym.iloc[0]

        signal = row.get("signal", "none")
        score_summary = row.get("score_summary", "")
        mtf_long = row.get("mtf_long_score", np.nan)
        mtf_short = row.get("mtf_short_score", np.nan)

        st.markdown(f"#### {sym} – Multi-Timeframe Summary")
        st.markdown(
            f"- **Signal**: `{signal}`  \n"
            f"- **Scores**: {score_summary} "
            f"(L={_fmt_val(mtf_long)}, S={_fmt_val(mtf_short)})"
        )

        etf_lines = []
        if "etf_symbol_primary" in row.index:
            etf_lines.append(
                f"- **Primary ETF**: {row.get('etf_symbol_primary', '—')} "
                f"(L={_fmt_val(row.get('etf_primary_long_score'))}, "
                f"S={_fmt_val(row.get('etf_primary_short_score'))})"
            )
        if "etf_symbol_secondary" in row.index:
            etf_lines.append(
                f"- **Secondary ETF**: {row.get('etf_symbol_secondary', '—')} "
                f"(L={_fmt_val(row.get('etf_secondary_long_score'))}, "
                f"S={_fmt_val(row.get('etf_secondary_short_score'))})"
            )

        if etf_lines:
            st.markdown("**ETF Trend Overlay:**")
            st.markdown("\n".join(etf_lines))

        st.markdown("---")

        col_l, col_m, col_u = st.columns(3)
        _render_tf_block(col_l, row, prefix="lower", title="Lower TF")
        _render_tf_block(col_m, row, prefix="middle", title="Middle TF")
        _render_tf_block(col_u, row, prefix="upper", title="Upper TF")

        with st.expander("Raw row (all columns)", expanded=False):
            st.write(df_sym.T)
