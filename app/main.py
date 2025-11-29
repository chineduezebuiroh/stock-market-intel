import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Market Intel – MTF Dashboard",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_parquet_safe(path: Path) -> pd.DataFrame | None:
    """
    Load a parquet file if it exists, otherwise return None.
    """
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return None


def render_snapshot_tab(namespace: str, timeframe: str, title: str):
    """
    Render a single-timeframe snapshot table.
    Expects data/snapshot_{namespace}_{timeframe}.parquet
    """
    st.subheader(title)
    p = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
    df = load_parquet_safe(p)

    if df is None or df.empty:
        st.info(f"No snapshot data found for `{namespace}:{timeframe}`.")
        st.code(str(p))
        return

    # Make sure symbol is visible as a column, not buried in the index
    if "symbol" in df.columns:
        df_display = df.copy()
    else:
        df_display = df.reset_index()

    st.dataframe(df_display, use_container_width=True)

"""
def render_combo_tab_stocks_c_dwm_shortlist():
    
    #Render the combo for:
    #  - namespace: stocks
    #  - combo: stocks_c_dwm_shortlist
    #  - file: data/combo_stocks_c_dwm_shortlist.parquet
    
    st.subheader("Stocks C: Daily / Weekly / Monthly – Shortlist")

    combo_path = DATA / "combo_stocks_c_dwm_shortlist.parquet"
    df = load_parquet_safe(combo_path)

    if df is None or df.empty:
        st.info("No combo data found for `stocks_c_dwm_shortlist` yet.")
        st.code(str(combo_path))
        return

    # Ensure we have symbol as a visible column
    if "symbol" not in df.columns:
        df = df.reset_index()

    # Optional: basic signal filter
    if "signal" in df.columns:
        signal_choice = st.radio(
            "Filter by signal:",
            options=["all", "long", "short", "watch"],
            horizontal=True,
        )

        if signal_choice != "all":
            df = df[df["signal"] == signal_choice]

    # For now, just show everything. Later we can:
    #  - Hide raw indicator columns
    #  - Add sector / industry / ETF trend columns
    st.dataframe(df, use_container_width=True)
"""


def render_combo_tab_stocks_c_dwm_shortlist():
    """
    Render the combo for:
      - namespace: stocks
      - combo: stocks_c_dwm_shortlist
      - file: data/combo_stocks_c_dwm_shortlist.parquet
    """
    st.subheader("Stocks C: Daily / Weekly / Monthly – Shortlist")

    combo_path = DATA / "combo_stocks_c_dwm_shortlist.parquet"
    df = load_parquet_safe(combo_path)

    if df is None or df.empty:
        st.info("No combo data found for `stocks_c_dwm_shortlist` yet.")
        st.code(str(combo_path))
        return

    # Ensure symbol column
    if "symbol" not in df.columns:
        df = df.reset_index()

    # Optional: basic signal filter
    if "signal" in df.columns:
        signal_choice = st.radio(
            "Filter by signal:",
            options=["all", "long", "short", "watch"],
            horizontal=True,
        )
        if signal_choice != "all":
            df = df[df["signal"] == signal_choice]

    if df.empty:
        st.info("No rows match the selected signal filter.")
        return

    # Curated view: price/volume + EMAs for each timeframe
    preferred_cols = [
        "symbol",
        "signal",

        # Lower (daily)
        "lower_close", "lower_volume",
        "lower_ema_8", "lower_ema_21",
        "lower_atr_14",

        # Middle (weekly)
        "middle_close", "middle_volume",
        "middle_ema_8", "middle_ema_21",

        # Upper (monthly)
        "upper_close", "upper_volume",
        "upper_ema_8", "upper_ema_21",
    ]

    existing_cols = [c for c in preferred_cols if c in df.columns]
    df_view = df[existing_cols].copy()

    # Sort by symbol for now (later we might sort by signal strength, etc.)
    df_view = df_view.sort_values(by="symbol")

    st.dataframe(df_view, use_container_width=True)



# -----------------------------------------------------------------------------
# Layout / Tabs
# -----------------------------------------------------------------------------
st.title("Stock Market Intel – Multi-Timeframe Dashboard")

tabs = st.tabs(
    [
        "Daily snapshot (stocks)",
        "Weekly snapshot (stocks)",
        "Monthly snapshot (stocks)",
        "Stocks C: D/W/M Shortlist",
    ]
)

with tabs[0]:
    render_snapshot_tab("stocks", "daily", "Stocks – Daily Snapshot")

with tabs[1]:
    render_snapshot_tab("stocks", "weekly", "Stocks – Weekly Snapshot")

with tabs[2]:
    render_snapshot_tab("stocks", "monthly", "Stocks – Monthly Snapshot")

with tabs[3]:
    render_combo_tab_stocks_c_dwm_shortlist()
