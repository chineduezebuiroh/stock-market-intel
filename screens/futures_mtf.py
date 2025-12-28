from __future__ import annotations

# screens/futures_mtf.py

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
from app.field_catalog import FUTURES_BASE_COLS
from screens.style_helpers import apply_signal_row_styles
from screens.debug_panel import render_symbol_debug_panel


FUTURES_COMBOS = [
    ("Futures 1 · 1h / 4h / D", "futures_1_1h4hd_shortlist"),
    ("Futures 2 · 4h / D / W",  "futures_2_4hdw_shortlist"),
    ("Futures 3 · D / W / M",   "futures_3_dwm_shortlist"),
]


def render_futures_mtf_tab():
    st.subheader("Futures – Multi-Timeframe Combos")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        combo_label = st.selectbox(
            "Futures combo",
            options=[label for (label, _) in FUTURES_COMBOS],
            index=0,
            key="futures_combo_select",
        )
    with col_right:
        signal_filter = st.selectbox(
            "Signal filter",
            options=["all", "long", "short", "watch", "anti", "none"],
            index=0,
            key="futures_signal_filter",
        )

    label_to_name = {label: name for (label, name) in FUTURES_COMBOS}
    combo_name = label_to_name[combo_label]

    df = load_combo_safe(combo_name)
    if df.empty:
        st.info(f"No data for `{combo_name}`. Run jobs/run_combo.py futures {combo_name}.")
        return

    df_full = apply_signal_filter(df, signal_filter)
    df_full = sort_for_view(df_full, signal_filter)
    if df_full.empty:
        st.info("No rows match the selected signal filter.")
        return

    df_full = add_score_summary(df_full)
    df_view_flat = select_display_cols(df_full, FUTURES_BASE_COLS, include_time_cols=True)

    # ✅ new collapsible groups
    df_view_flat = filter_combo_columns_ui(df_view_flat, key_prefix="futures_cols")

    df_view = add_mtf_grouped_headers(df_view_flat)
    display_obj = apply_signal_row_styles(df_view)

    st.dataframe(display_obj, use_container_width=True, hide_index=True)
    st.caption(f"Rows: {len(df_view)} · Combo: `{combo_name}`")

    st.markdown("---")
    render_symbol_debug_panel(
        df_full,
        label=f"Inspect futures symbol – {combo_label}",
        key="futures_debug_symbol",
    )
