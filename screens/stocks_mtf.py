from __future__ import annotations

# screens/stocks_mtf.py

import streamlit as st

from app.data_access import load_combo_safe
from app.combo_view import (
    apply_signal_filter,
    sort_for_view,
    add_score_summary,
    select_display_cols,
    add_mtf_grouped_headers,
    filter_combo_columns_ui,
)
from app.field_catalog import STOCKS_BASE_COLS, STOCKS_OPTIONS_EXTRA_COLS
from screens.style_helpers import style_etf_scores, apply_signal_row_styles
from screens.debug_panel import render_symbol_debug_panel


STOCK_COMBOS = [
    ("130m/D/W – Shortlist",      "stocks_d_130mdw_shortlist", "shortlist"),
    ("D/W/M – Shortlist",         "stocks_c_dwm_shortlist",  "shortlist"),
    ("D/W/M – Options-eligible",  "stocks_c_dwm_all",        "options"),
    ("W/M/Q – Shortlist",         "stocks_b_wmq_shortlist",  "shortlist"),
    ("W/M/Q – Options-eligible",  "stocks_b_wmq_all",        "options"),
    ("M/Q/Y – Shortlist",         "stocks_a_mqy_shortlist",  "shortlist"),
    ("M/Q/Y – Options-eligible",  "stocks_a_mqy_all",        "options"),
]


def render_stocks_mtf_tab():
    st.subheader("Stocks – Multi-Timeframe Combos")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        combo_label = st.selectbox(
            "Combo family",
            options=[label for (label, _, _) in STOCK_COMBOS],
            index=0,
            key="stocks_combo_select",
        )
    with col_right:
        signal_filter = st.selectbox(
            "Signal filter",
            options=["all", "long", "short", "watch", "anti"],
            index=0,
            key="stocks_signal_filter",
        )

    label_to_cfg = {label: (name, universe) for (label, name, universe) in STOCK_COMBOS}
    combo_name, universe_type = label_to_cfg[combo_label]

    df = load_combo_safe(combo_name)
    if df.empty:
        st.info(f"No data for `{combo_name}`. Run jobs/run_combo.py stocks {combo_name}.")
        return

    df_full = apply_signal_filter(df, signal_filter)
    df_full = sort_for_view(df_full, signal_filter)
    if df_full.empty:
        st.info("No rows match the selected signal filter.")
        return

    df_full = add_score_summary(df_full)

    base_cols = STOCKS_BASE_COLS.copy()
    if universe_type == "options":
        base_cols += STOCKS_OPTIONS_EXTRA_COLS

    df_view_flat = select_display_cols(df_full, base_cols, include_time_cols=True)

    # ✅ new collapsible groups (show all by default)
    df_view_flat = filter_combo_columns_ui(df_view_flat, key_prefix="stocks_cols")

    df_view = add_mtf_grouped_headers(df_view_flat)

    display_obj = df_view
    if universe_type == "options" and not df_view.empty:
        display_obj = style_etf_scores(df_view)
    display_obj = apply_signal_row_styles(display_obj)

    st.dataframe(display_obj, use_container_width=True, hide_index=True)
    st.caption(f"Rows: {len(df_view)} · Combo: `{combo_name}` · Universe: {universe_type}")

    st.markdown("---")
    render_symbol_debug_panel(
        df_full,
        label=f"Inspect symbol – {combo_label}",
        key="stocks_debug_symbol",
    )
