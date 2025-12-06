import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import numpy as np

# =============================================================================
# Paths
# =============================================================================
DATA = ROOT / "data"

from screens.style_helpers import style_etf_scores, apply_signal_row_styles

# =============================================================================
# Streamlit page config
# =============================================================================
st.set_page_config(
    page_title="Stock Market Intel – MTF Dashboard",
    layout="wide",
)

# =============================================================================
# Generic combo helpers (stocks + futures)
# =============================================================================
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


def load_combo_safe(combo_name: str) -> pd.DataFrame:
    """
    Load data/combo_<combo_name>.parquet, or return empty DataFrame.
    """
    path = DATA / f"combo_{combo_name}.parquet"
    df = load_parquet_safe(path)
    if df is None:
        return pd.DataFrame()

    # Ensure core signal columns exist
    for col in ["symbol", "signal", "signal_side", "mtf_long_score", "mtf_short_score"]:
        if col not in df.columns:
            df[col] = np.nan

    # Backfill signal_side if missing
    if "signal_side" in df.columns and "signal" in df.columns:
        df["signal_side"] = df["signal_side"].fillna(df["signal"])

    return df


def apply_signal_filter(df: pd.DataFrame, signal_filter: str) -> pd.DataFrame:
    if df.empty or signal_filter == "all":
        return df

    col = "signal" if "signal" in df.columns else "signal_side"
    if col not in df.columns:
        return df

    return df[df[col] == signal_filter]


def sort_for_view(df: pd.DataFrame, signal_filter: str) -> pd.DataFrame:
    if df.empty:
        return df

    if signal_filter == "long" and "mtf_long_score" in df.columns:
        return df.sort_values("mtf_long_score", ascending=False, na_position="last")
    if signal_filter == "short" and "mtf_short_score" in df.columns:
        return df.sort_values("mtf_short_score", ascending=False, na_position="last")

    if {"mtf_long_score", "mtf_short_score"}.issubset(df.columns):
        df = df.copy()
        df["mtf_abs_max"] = df[["mtf_long_score", "mtf_short_score"]].abs().max(axis=1)
        df = df.sort_values("mtf_abs_max", ascending=False, na_position="last")
        df = df.drop(columns=["mtf_abs_max"])
    return df


def add_score_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable score summary column combining 
    mtf_long_score and mtf_short_score as 'L/S'.

    Keeps original numeric score columns untouched.
    """
    if df is None or df.empty:
        return df

    if "mtf_long_score" not in df.columns or "mtf_short_score" not in df.columns:
        return df

    df = df.copy()

    def _fmt(row):
        ls = row.get("mtf_long_score", np.nan)
        ss = row.get("mtf_short_score", np.nan)
        try:
            if pd.isna(ls) and pd.isna(ss):
                return ""
            return f"{float(ls):.1f} / {float(ss):.1f}"
        except Exception:
            return ""

    df["score_summary"] = df.apply(_fmt, axis=1)
    return df
    

def add_mtf_grouped_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert flat columns like 'lower_close', 'middle_ema_8', 'upper_wyckoff_stage'
    into a MultiIndex with top-level groups: META / LOWER / MIDDLE / UPPER.

    Example:
        'symbol' -> ('META', 'symbol')
        'lower_close' -> ('LOWER', 'close')
        'middle_ema_8' -> ('MIDDLE', 'ema_8')
        'upper_wyckoff_stage' -> ('UPPER', 'wyckoff_stage')
    """
    if df is None or df.empty:
        return df

    new_cols = []
    for c in df.columns:
        # we only expect plain strings here
        name = str(c)

        if name.startswith("lower_"):
            new_cols.append(("LOWER", name.replace("lower_", "")))
        elif name.startswith("middle_"):
            new_cols.append(("MIDDLE", name.replace("middle_", "")))
        elif name.startswith("upper_"):
            new_cols.append(("UPPER", name.replace("upper_", "")))
        else:
            new_cols.append(("META", name))

    out = df.copy()
    out.columns = pd.MultiIndex.from_tuples(new_cols)
    return out

# =============================================================================
# UNIVERSAL SYMBOL DEBUG PANEL
# =============================================================================
def render_symbol_debug_panel(
    df: pd.DataFrame,
    label: str = "Debug panel",
    key: str = "debug_symbol_select",
):
    """
    Renders a dropdown to pick a symbol and shows the full row of all
    columns (lower_*, middle_*, upper_*, scores, etc.) for QA.

    df should already be filtered/sorted to match what the user is seeing
    in the main table, but should *not* be column-trimmed.
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return

    st.markdown(f"#### {label}")

    symbols = sorted(df["symbol"].dropna().unique().tolist())
    if not symbols:
        st.info("No symbols available for debug.")
        return

    selected = st.selectbox(
        "Select symbol to inspect",
        options=symbols,
        key=key,
    )

    row_df = df[df["symbol"] == selected]
    if row_df.empty:
        st.info("No data for selected symbol.")
        return

    # Take the first row (they should all be identical for a symbol).
    row = row_df.iloc[0]

    # Transpose for readability: one column with all fields
    row_t = row.to_frame(name=selected)

    st.dataframe(
        row_t,
        use_container_width=True,
        hide_index=False,
    )




# =============================================================================
# Stocks MTF unified view (all combos)
# =============================================================================
STOCK_COMBOS = [
    ("130m/D/W – Shortlist",      "stocks_d_130mdw_shortlist", "shortlist"),
    ("D/W/M – Shortlist",         "stocks_c_dwm_shortlist",  "shortlist"),
    ("D/W/M – Options-eligible",  "stocks_c_dwm_all",        "options"),
    ("W/M/Q – Shortlist",         "stocks_b_wmq_shortlist",  "shortlist"),
    ("W/M/Q – Options-eligible",  "stocks_b_wmq_all",        "options"),
    ("M/Q/Y – Shortlist",         "stocks_a_mqy_shortlist",  "shortlist"),
    ("M/Q/Y – Options-eligible",  "stocks_a_mqy_all",        "options"),
]


def select_display_cols_stocks(df: pd.DataFrame, universe_type: str) -> pd.DataFrame:
    if df.empty:
        return df

    base_cols = [
        "symbol",
        "signal",
        "score_summary",        # <-- NEW: right after signal
        "mtf_long_score",
        "mtf_short_score",
        # lower
        "lower_wyckoff_stage",
        "lower_exh_abs_pa_current_bar",
        "lower_exh_abs_pa_prior_bar",
        "lower_significant_volume",
        "lower_spy_qqq_vol_ma_ratio",
        "lower_ma_trend_cloud",
        "lower_macdv_core",
        "lower_ttm_squeeze_pro",
        # middle
        "middle_wyckoff_stage",
        "middle_exh_abs_pa_prior_bar",
        "middle_significant_volume",
        "middle_spy_qqq_vol_ma_ratio",
        # upper
        "upper_wyckoff_stage",
        "upper_exh_abs_pa_prior_bar",
    ]

    if universe_type == "options":
        base_cols += [
            "etf_symbol_primary",
            "etf_primary_long_score",
            "etf_primary_short_score",
            "etf_symbol_secondary",
            "etf_secondary_long_score",
            "etf_secondary_short_score",
        ]

    cols = [c for c in base_cols if c in df.columns]
    return df[cols] if cols else df


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
            options=["all", "long", "short", "watch"],
            index=0,
            key="stocks_signal_filter",
        )

    label_to_cfg = {label: (name, universe) for (label, name, universe) in STOCK_COMBOS}
    combo_name, universe_type = label_to_cfg[combo_label]

    df = load_combo_safe(combo_name)
    if df.empty:
        st.info(f"No data for `{combo_name}`. Run jobs/run_combo.py stocks {combo_name}.")
        return
    
    # Apply signal filter + sorting on the *full* frame first
    df_full = apply_signal_filter(df, signal_filter)
    df_full = sort_for_view(df_full, signal_filter)

    if df_full.empty:
        st.info("No rows match the selected signal filter.")
        return

    # Add score summary for display
    df_full = add_score_summary(df_full)
    
    # Trim for main display (plain DataFrame)
    df_view = select_display_cols_stocks(df_full, universe_type)
    
    # --- Styling chain: ETF styling (options only) -> signal row shading ---
    display_obj = df_view  # can be DataFrame or Styler

    if universe_type == "options" and not df_view.empty:
        # style_etf_scores returns a Styler
        display_obj = style_etf_scores(df_view)

    # Row shading for all universes (works with DataFrame or Styler)
    display_obj = apply_signal_row_styles(display_obj)

    st.dataframe(
        display_obj,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Rows: {len(df_view)} · Combo: `{combo_name}` · Universe: {universe_type}"
    )

    # --- Symbol-level debug panel (full row, not trimmed/styled view) ---
    st.markdown("---")
    render_symbol_debug_panel(
        df_full,
        label=f"Inspect symbol – {combo_label}",
        key="stocks_debug_symbol",
    )

# =============================================================================
# Futures MTF view
# =============================================================================
FUTURES_COMBOS = [
    ("Futures 1 · 1h / 4h / D", "futures_1_1h4hd_shortlist"),
    ("Futures 2 · 4h / D / W",  "futures_2_4hdw_shortlist"),
    ("Futures 3 · D / W / M",   "futures_3_dwm_shortlist"),
]


def select_display_cols_futures(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    preferred = [
        "symbol",
        "signal",
        "score_summary",        # <-- NEW
        "mtf_long_score",
        "mtf_short_score",
        # lower (1h / 4h / D depending on combo)
        "lower_wyckoff_stage",
        "lower_ma_trend_cloud",
        "lower_macdv_core",
        "lower_ttm_squeeze_pro",
        "lower_exh_abs_pa_current_bar",
        "lower_exh_abs_pa_prior_bar",
        "lower_significant_volume",
        "lower_spy_qqq_vol_ma_ratio",
        # middle
        "middle_wyckoff_stage",
        "middle_ma_trend_cloud",
        "middle_macdv_core",
        "middle_exh_abs_pa_prior_bar",
        "middle_significant_volume",
        # upper
        "upper_wyckoff_stage",
        "upper_exh_abs_pa_prior_bar",
    ]

    cols = [c for c in preferred if c in df.columns]
    return df[cols] if cols else df


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
            options=["all", "long", "short", "watch", "none"],
            index=0,
            key="futures_signal_filter",
        )

    label_to_name = {label: name for (label, name) in FUTURES_COMBOS}
    combo_name = label_to_name[combo_label]

    df = load_combo_safe(combo_name)
    if df.empty:
        st.info(f"No data for `{combo_name}`. Run jobs/run_combo.py futures {combo_name}.")
        return

    # Apply signal filter + sorting on full frame
    df_full = apply_signal_filter(df, signal_filter)
    df_full = sort_for_view(df_full, signal_filter)

    if df_full.empty:
        st.info("No rows match the selected signal filter.")
        return

    # Add score summary
    df_full = add_score_summary(df_full)
    
    # Trim for main futures view
    df_view = select_display_cols_futures(df_full)

    # Apply row shading based on signal
    display_obj = apply_signal_row_styles(df_view)

    st.dataframe(
        display_obj,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Rows: {len(df_view)} · Combo: `{combo_name}`"
    )

    # --- Symbol-level debug panel for futures ---
    st.markdown("---")
    render_symbol_debug_panel(
        df_full,
        label=f"Inspect futures symbol – {combo_label}",
        key="futures_debug_symbol",
    )


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

# =============================================================================
# Layout / Tabs
# =============================================================================
st.title("Stock Market Intel – Multi-Timeframe Dashboard")
"""
tab_daily, tab_weekly, tab_monthly, tab_130mdw, tab_dwm, tab_wmq, tab_mqy = st.tabs(
    [
        "Stocks – Daily",
        "Stocks – Weekly",
        "Stocks – Monthly",
        "Stocks – 130m/D/W (Combos)",
        "Stocks – D/W/M (Combos)",
        "Stocks – W/M/Q (Combos)",
        "Stocks – M/Q/Y (Combos)",
    ]
)
"""

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



"""
with tab_130mdw:
    st.subheader("Stocks – 130m/D/W Multi-Timeframe Combos")

    # Shared signal filter for both 130mDW tables
    intra130mdw_signal_filter = st.radio(
        "Filter by signal:",
        options=["all", "long", "short", "watch"],
        index=0,
        horizontal=True,
        key="intra130mdw_signal_filter",
    )

    # ---------- 130mDW Shortlist ----------
    st.markdown("### Shortlist universe (130m/D/W combo)")
    p_intra130mdw_short = DATA / "combo_stocks_d_130mdw_shortlist.parquet"
    if p_intra130mdw_short.exists():
        df_intra130mdw_short = pd.read_parquet(p_intra130mdw_short)

        # Apply signal filter
        if "signal" in df_intra130mdw_short.columns and intra130mdw_signal_filter != "all":
            df_intra130mdw_short = df_intra130mdw_short[df_intra130mdw_short["signal"] == intra130mdw_signal_filter]

        # Sort by scores
        if {"mtf_long_score", "mtf_short_score"}.issubset(df_intra130mdw_short.columns):
            df_intra130mdw_short = df_intra130mdw_short.sort_values(
                ["mtf_long_score", "mtf_short_score"], ascending=[False, False]
            )

        # Choose key columns to display
        intra130mdw_cols_short = [
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
        intra130mdw_existing_short = [c for c in intra130mdw_cols_short if c in df_intra130mdw_short.columns]
        st.dataframe(df_intra130mdw_short[intra130mdw_existing_short])
    else:
        st.info("130mDW shortlist combo not found. Run jobs/run_combo.py stocks stocks_d_130mdw_shortlist")

    st.markdown("---")


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

        st.dataframe(
            df_short[existing_cols_short],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Shortlist D/W/M combo not found. "
            "Run jobs/run_combo.py stocks stocks_c_dwm_shortlist"
        )

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
            # ETF guardrail columns
            "etf_symbol_primary",
            "etf_primary_long_score",
            "etf_primary_short_score",
            "etf_symbol_secondary",
            "etf_secondary_long_score",
            "etf_secondary_short_score",
        ]

        # Only keep columns that actually exist
        existing_cols_opts = [c for c in cols_opts if c in df_opts.columns]

        opt_view = df_opts[existing_cols_opts]

        # Apply ETF styling
        styled_options = style_etf_scores(opt_view)

        st.dataframe(
            styled_options,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "Options D/W/M combo not found. "
            "Run jobs/run_combo.py stocks stocks_c_dwm_all"
        )



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
    p_wmq_short = DATA / "combo_stocks_b_wmq_shortlist.parquet"
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
    p_wmq_all = DATA / "combo_stocks_b_wmq_all.parquet"
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


with tab_mqy:
    st.subheader("Stocks – M/Q/Y Multi-Timeframe Combos")

    # Shared signal filter for both WMQ tables
    mqy_signal_filter = st.radio(
        "Filter by signal:",
        options=["all", "long", "short", "watch"],
        index=0,
        horizontal=True,
        key="mqy_signal_filter",
    )

    # ---------- MQY Shortlist ----------
    st.markdown("### Shortlist universe (M/Q/Y combo)")
    p_mqy_short = DATA / "combo_stocks_a_mqy_shortlist.parquet"
    if p_mqy_short.exists():
        df_mqy_short = pd.read_parquet(p_mqy_short)

        # Apply signal filter
        if "signal" in df_mqy_short.columns and mqy_signal_filter != "all":
            df_mqy_short = df_mqy_short[df_mqy_short["signal"] == mqy_signal_filter]

        # Sort by scores
        if {"mtf_long_score", "mtf_short_score"}.issubset(df_mqy_short.columns):
            df_mqy_short = df_mqy_short.sort_values(
                ["mtf_long_score", "mtf_short_score"], ascending=[False, False]
            )

        # Choose key columns to display
        mqy_cols_short = [
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
        mqy_existing_short = [c for c in mqy_cols_short if c in df_mqy_short.columns]
        st.dataframe(df_mqy_short[mqy_existing_short])
    else:
        st.info("MQY shortlist combo not found. Run jobs/run_combo.py stocks stocks_a_mqy_shortlist")

    st.markdown("---")

    # ---------- MQY Options-eligible ----------
    st.markdown("### Options-eligible universe (W/M/Q combo)")
    p_mqy_all = DATA / "combo_stocks_a_mqy_all.parquet"
    if p_mqy_all.exists():
        df_mqy_all = pd.read_parquet(p_mqy_all)

        # Apply signal filter
        if "signal" in df_mqy_all.columns and mqy_signal_filter != "all":
            df_mqy_all = df_mqy_all[df_mqy_all["signal"] == mqy_signal_filter]

        # Sort by scores
        if {"mtf_long_score", "mtf_short_score"}.issubset(df_mqy_all.columns):
            df_mqy_all = df_mqy_all.sort_values(
                ["mtf_long_score", "mtf_short_score"], ascending=[False, False]
            )

        mqy_cols_all = [
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
        mqy_existing_all = [c for c in mqy_cols_all if c in df_mqy_all.columns]
        st.dataframe(df_mqy_all[mqy_existing_all])
    else:
        st.info("MQY options combo not found. Run jobs/run_combo.py stocks stocks_a_mqy_all")
"""

with tab_stocks_mtf:
    render_stocks_mtf_tab()

with tab_futures_mtf:
    render_futures_mtf_tab()
