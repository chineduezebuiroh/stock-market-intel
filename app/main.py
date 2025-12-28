from __future__ import annotations

# app/main.py

import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from screens.stocks_mtf import render_stocks_mtf_tab
from screens.futures_mtf import render_futures_mtf_tab
from screens.snapshots import render_snapshot_tab


st.set_page_config(
    page_title="Stock Market Intel – MTF Dashboard",
    layout="wide",
)

st.title("Stock Market Intel – Multi-Timeframe Dashboard")

tab_daily, tab_weekly, tab_monthly, tab_stocks_mtf, tab_futures_mtf = st.tabs(
    [
        "Stocks – Daily",
        "Stocks – Weekly",
        "Stocks – Monthly",
        "Stocks – MTF Combos",
        "Futures – MTF Combos",
    ]
)

with tab_daily:
    render_snapshot_tab("stocks", "daily", "Stocks – Daily Snapshot")

with tab_weekly:
    render_snapshot_tab("stocks", "weekly", "Stocks – Weekly Snapshot")

with tab_monthly:
    render_snapshot_tab("stocks", "monthly", "Stocks – Monthly Snapshot")

with tab_stocks_mtf:
    render_stocks_mtf_tab()

with tab_futures_mtf:
    render_futures_mtf_tab()
