from pathlib import Path
import os  # NEW
import sys
"""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""
import pandas as pd
import streamlit as st
import numpy as np

from screens.style_helpers import style_etf_scores, apply_signal_row_styles

from core.paths import ROOT, DATA #, CFG, REF  # import what you need

# =============================================================================
# Paths / Constants
# =============================================================================
"""
DATA_DIR = os.getenv("DATA_DIR", "data")  # default to 'data'
DATA = ROOT / DATA_DIR
"""
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
def _fmt_val(v):
    """Human-friendly formatting for debug panel values."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        # For small ints like -2..2, show as int; otherwise one decimal.
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return f"{v:.2f}"
        return str(v)
    except Exception:
        return str(v)


def _render_tf_block(col_container, row: pd.Series, prefix: str, title: str):
    """
    Render a single timeframe block (lower/middle/upper) in the debug panel.
    prefix: 'lower', 'middle', or 'upper'
    """
    fields = [
        ("close", "Close"),
        ("volume", "Volume"),
        ("wyckoff_stage", "Wyckoff Stage"),
        ("exh_abs_pa_current_bar", "Exh/Abs (current)"),
        ("exh_abs_pa_prior_bar", "Exh/Abs (prior)"),
        ("sig_vol_current_bar", "Sig Vol (current)"),
        ("sig_vol_prior_bar", "Sig Vol (prior)"),
        ("spy_qqq_vol_ma_ratio", "SPY/QQQ Vol Ratio"),
        ("ma_trend_bullish", "MA Trend Bull"),
        ("ma_trend_bearish", "MA Trend Bear"),
        ("macdv_core", "MACDV Core"),
        ("ttm_squeeze_pro", "TTM Squeeze Pro"),
        ("ema_8", "EMA 8"),
        ("ema_21", "EMA 21"),
        ("atr_14", "ATR 14"),
    ]

    col_container.markdown(f"### {title}")

    lines = []
    for col_suffix, label in fields:
        col_name = f"{prefix}_{col_suffix}"
        if col_name in row.index:
            val = row[col_name]
            lines.append(f"- **{label}**: {_fmt_val(val)}")

    if lines:
        col_container.markdown("\n".join(lines))
    else:
        col_container.info("No fields available for this timeframe.")


def render_symbol_debug_panel(
    df_full: pd.DataFrame,
    label: str = "Inspect symbol",
    key: str | None = None,
):
    """
    Symbol-level deep dive panel:

    - Select a symbol from the current combo.
    - Show meta summary (signal, scores, ETF guardrails if present).
    - Show three columns: LOWER / MIDDLE / UPPER timeframe blocks.
    """
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

        # Assume one row per symbol in combo output
        row = df_sym.iloc[0]

        # ------------------------------------------------------------------
        # Meta summary
        # ------------------------------------------------------------------
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

        # ETF overlay if present (options-eligible stocks)
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

        # ------------------------------------------------------------------
        # Three-column MTF breakdown
        # ------------------------------------------------------------------
        col_l, col_m, col_u = st.columns(3)

        _render_tf_block(col_l, row, prefix="lower", title="Lower TF")
        _render_tf_block(col_m, row, prefix="middle", title="Middle TF")
        _render_tf_block(col_u, row, prefix="upper", title="Upper TF")

        # Optional: raw row toggle for deep debugging
        with st.expander("Raw row (all columns)", expanded=False):
            st.write(df_sym.T)

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
        "lower_sig_vol_current_bar",
        "lower_sig_vol_prior_bar",
        "lower_spy_qqq_vol_ma_ratio",
        "lower_ma_trend_bullish",
        "lower_ma_trend_bearish",
        "lower_macdv_core",
        "lower_ttm_squeeze_pro",
        # middle
        "middle_wyckoff_stage",
        "middle_exh_abs_pa_prior_bar",
        "middle_sig_vol_current_bar",
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
    df_view_flat = select_display_cols_stocks(df_full, universe_type)

    # Add grouped headers (META / LOWER / MIDDLE / UPPER)
    df_view = add_mtf_grouped_headers(df_view_flat)
    
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
        # lower
        "lower_wyckoff_stage",
        "lower_exh_abs_pa_current_bar",
        "lower_exh_abs_pa_prior_bar",
        "lower_sig_vol_current_bar",
        "lower_sig_vol_prior_bar",
        "lower_spy_qqq_vol_ma_ratio",
        "lower_ma_trend_bullish",
        "lower_ma_trend_bearish",
        "lower_macdv_core",
        "lower_ttm_squeeze_pro",
        # middle
        "middle_wyckoff_stage",
        "middle_exh_abs_pa_prior_bar",
        "middle_sig_vol_current_bar",
        "middle_spy_qqq_vol_ma_ratio",
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
    df_view_flat = select_display_cols_futures(df_full)

    # Add grouped headers
    df_view = add_mtf_grouped_headers(df_view_flat)

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

with tab_stocks_mtf:
    render_stocks_mtf_tab()

with tab_futures_mtf:
    render_futures_mtf_tab()
