from __future__ import annotations

# screens/snapshots.py

import streamlit as st

from core.paths import DATA
from app.data_access import load_parquet_safe


def render_snapshot_tab(namespace: str, timeframe: str, title: str):
    st.subheader(title)
    p = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
    df = load_parquet_safe(p)

    if df is None or df.empty:
        st.info(f"No snapshot data found for `{namespace}:{timeframe}`.")
        st.code(str(p))
        return

    if "symbol" in df.columns:
        df_display = df.copy()
    else:
        df_display = df.reset_index()

    st.dataframe(df_display, use_container_width=True, hide_index=True)


def render_stock_snapshots_tabs():
    # Call this inside each tab body to keep main.py tiny
    render_snapshot_tab("stocks", "daily", "Stocks – Daily Snapshot")
    # You can add sorting here if you want, but keep it light.
