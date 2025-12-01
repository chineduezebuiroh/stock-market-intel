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

tab_daily, tab_weekly, tab_monthly, tab_dwm = st.tabs(
    [
        "Stocks – Daily",
        "Stocks – Weekly",
        "Stocks – Monthly",
        "Stocks – D/W/M (Combos)",
        "Stocks – W/M/Q (Combos)",
    ]
)


with tab_daily:
    st.subheader("Stocks – Daily Snapshot")
    p = DATA / "snapshot_stocks_daily.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # optional: sort by symbol
        df = df.sort_values("symbol")
        st.dataframe(df)
    else:
        st.info("Daily snapshot not found. Run jobs/run_timeframe.py stocks daily --cascade")

with tab_weekly:
    st.subheader("Stocks – Weekly Snapshot")
    p = DATA / "snapshot_stocks_weekly.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df = df.sort_values("symbol")
        st.dataframe(df)
    else:
        st.info("Weekly snapshot not found. Run jobs/run_timeframe.py stocks daily --cascade")

with tab_monthly:
    st.subheader("Stocks – Monthly Snapshot")
    p = DATA / "snapshot_stocks_monthly.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df = df.sort_values("symbol")
        st.dataframe(df)
    else:
        st.info("Monthly snapshot not found. Run jobs/run_timeframe.py stocks daily --cascade")


with tab_dwm:
    st.subheader("Stocks – D/W/M Multi-Timeframe Combos")

    # Shared signal filter for both tables
    signal_filter = st.radio(
        "Filter by signal:",
        options=["all", "long", "short", "watch"],
        index=0,
        horizontal=True,
        key="dwm_signal_filter",
    )

    # ---------- Shortlist D/W/M ----------
    st.markdown("### Shortlist universe (D/W/M combo)")
    p_shortlist = DATA / "combo_stocks_c_dwm_shortlist.parquet"
    if p_shortlist.exists():
        df_short = pd.read_parquet(p_shortlist)

        # Apply signal filter if the column exists
        if "signal" in df_short.columns and signal_filter != "all":
            df_short = df_short[df_short["signal"] == signal_filter]

        # Example: sort by long score desc, then short score desc
        if {"mtf_long_score", "mtf_short_score"}.issubset(df_short.columns):
            df_short = df_short.sort_values(
                ["mtf_long_score", "mtf_short_score"], ascending=[False, False]
            )

        cols_short = [
            "symbol",
            "signal",
            "mtf_long_score",
            "mtf_short_score",
            "lower_wyckoff_stage",
            "lower_exh_abs_pa_current_bar",
            "lower_exh_abs_pa_prior_bar",
            "lower_significant_volume",
            "lower_spy_qqq_vol_ma_ratio",
            "lower_ma_trend_cloud",
            "lower_macdv_core",
            "lower_ttm_squeeze_pro",
            "middle_wyckoff_stage",
            "middle_exh_abs_pa_prior_bar",
            "middle_significant_volume",
            "middle_spy_qqq_vol_ma_ratio",
            "upper_wyckoff_stage",
            "upper_exh_abs_pa_prior_bar",
        ]
        existing_cols_short = [c for c in cols_short if c in df_short.columns]

        st.dataframe(df_short[existing_cols_short])
    else:
        st.info("Shortlist D/W/M combo not found. Run jobs/run_combo.py stocks stocks_c_dwm_shortlist")

    st.markdown("---")


    # ---------- Options-eligible D/W/M ----------
    st.markdown("### Options-eligible universe (D/W/M combo)")
    p_opts = DATA / "combo_stocks_c_dwm_all.parquet"
    if p_opts.exists():
        df_opts = pd.read_parquet(p_opts)

        # Apply same signal filter
        if "signal" in df_opts.columns and signal_filter != "all":
            df_opts = df_opts[df_opts["signal"] == signal_filter]

        if {"mtf_long_score", "mtf_short_score"}.issubset(df_opts.columns):
            df_opts = df_opts.sort_values(
                ["mtf_long_score", "mtf_short_score"], ascending=[False, False]
            )

        cols_opts = [
            "symbol",
            "signal",
            "mtf_long_score",
            "mtf_short_score",
            "lower_wyckoff_stage",
            "lower_exh_abs_pa_current_bar",
            "lower_exh_abs_pa_prior_bar",
            "lower_significant_volume",
            "lower_spy_qqq_vol_ma_ratio",
            "lower_ma_trend_cloud",
            "lower_macdv_core",
            "lower_ttm_squeeze_pro",
            "middle_wyckoff_stage",
            "middle_exh_abs_pa_prior_bar",
            "middle_significant_volume",
            "middle_spy_qqq_vol_ma_ratio",
            "upper_wyckoff_stage",
            "upper_exh_abs_pa_prior_bar",
        ]
        existing_cols_opts = [c for c in cols_opts if c in df_opts.columns]

        st.dataframe(df_opts[existing_cols_opts])
    else:
        st.info("Options D/W/M combo not found. Run jobs/run_combo.py stocks stocks_c_dwm_all")


with tab_wmq:
    st.subheader("Stocks – W/M/Q Multi-Timeframe Combos")

    # Shared signal filter for both WMQ tables
    wmq_signal_filter = st.radio(
        "Filter by signal:",
        options=["all", "long", "short", "watch"],
        index=0,
        horizontal=True,
        key="wmq_signal_filter",
    )

    # ---------- WMQ Shortlist ----------
    st.markdown("### Shortlist universe (W/M/Q combo)")
    p_wmq_short = DATA / "combo_stocks_c_wmq_shortlist.parquet"
    if p_wmq_short.exists():
        df_wmq_short = pd.read_parquet(p_wmq_short)

        # Apply signal filter
        if "signal" in df_wmq_short.columns and wmq_signal_filter != "all":
            df_wmq_short = df_wmq_short[df_wmq_short["signal"] == wmq_signal_filter]

        # Sort by scores
        if {"mtf_long_score", "mtf_short_score"}.issubset(df_wmq_short.columns):
            df_wmq_short = df_wmq_short.sort_values(
                ["mtf_long_score", "mtf_short_score"], ascending=[False, False]
            )

        # Choose key columns to display
        wmq_cols_short = [
            "symbol",
            "signal",
            "mtf_long_score",
            "mtf_short_score",
            "lower_wyckoff_stage",
            "lower_exh_abs_pa_current_bar",
            "lower_exh_abs_pa_prior_bar",
            "lower_significant_volume",
            "lower_spy_qqq_vol_ma_ratio",
            "lower_ma_trend_cloud",
            "lower_macdv_core",
            "lower_ttm_squeeze_pro",
            "middle_wyckoff_stage",
            "middle_exh_abs_pa_prior_bar",
            "middle_significant_volume",
            "middle_spy_qqq_vol_ma_ratio",
            "upper_wyckoff_stage",
            "upper_exh_abs_pa_prior_bar",
        ]
        wmq_existing_short = [c for c in wmq_cols_short if c in df_wmq_short.columns]
        st.dataframe(df_wmq_short[wmq_existing_short])
    else:
        st.info("WMQ shortlist combo not found. Run jobs/run_combo.py stocks stocks_c_wmq_shortlist")

    st.markdown("---")

    # ---------- WMQ Options-eligible ----------
    st.markdown("### Options-eligible universe (W/M/Q combo)")
    p_wmq_all = DATA / "combo_stocks_c_wmq_all.parquet"
    if p_wmq_all.exists():
        df_wmq_all = pd.read_parquet(p_wmq_all)

        # Apply signal filter
        if "signal" in df_wmq_all.columns and wmq_signal_filter != "all":
            df_wmq_all = df_wmq_all[df_wmq_all["signal"] == wmq_signal_filter]

        # Sort by scores
        if {"mtf_long_score", "mtf_short_score"}.issubset(df_wmq_all.columns):
            df_wmq_all = df_wmq_all.sort_values(
                ["mtf_long_score", "mtf_short_score"], ascending=[False, False]
            )

        wmq_cols_all = [
            "symbol",
            "signal",
            "mtf_long_score",
            "mtf_short_score",
            "lower_wyckoff_stage",
            "lower_exh_abs_pa_current_bar",
            "lower_exh_abs_pa_prior_bar",
            "lower_significant_volume",
            "lower_spy_qqq_vol_ma_ratio",
            "lower_ma_trend_cloud",
            "lower_macdv_core",
            "lower_ttm_squeeze_pro",
            "middle_wyckoff_stage",
            "middle_exh_abs_pa_prior_bar",
            "middle_significant_volume",
            "middle_spy_qqq_vol_ma_ratio",
            "upper_wyckoff_stage",
            "upper_exh_abs_pa_prior_bar",
        ]
        wmq_existing_all = [c for c in wmq_cols_all if c in df_wmq_all.columns]
        st.dataframe(df_wmq_all[wmq_existing_all])
    else:
        st.info("WMQ options combo not found. Run jobs/run_combo.py stocks stocks_c_wmq_all")

