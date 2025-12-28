# screens/style_helpers.py
import pandas as pd
import numpy as np
from pandas.io.formats.style import Styler

"""
def style_etf_scores(df: pd.DataFrame) -> Styler:
"""
"""
    Shade ETF scores:
      - long_score > 0 => green-ish
      - short_score > 0 => red-ish
      - 0 / NaN => neutral
"""
"""
    def color_long(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v) or v < 4:
                styles.append("")
            elif v >= 6:
                styles.append("background-color: rgba(0, 170, 0, 0.25); color: #006400;")
            else:
                styles.append("background-color: rgba(0, 170, 0, 0.15); color: #006400;")
        return styles

    def color_short(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v) or v < 4:
                styles.append("")
            elif v >= 6:
                styles.append("background-color: rgba(220, 20, 60, 0.25); color: #8B0000;")
            else:
                styles.append("background-color: rgba(220, 20, 60, 0.15); color: #8B0000;")
        return styles

    styler = df.style

    long_cols = [
        c for c in (
            "etf_primary_long_score",
            "etf_secondary_long_score",
        ) if c in df.columns
    ]
    short_cols = [
        c for c in (
            "etf_primary_short_score",
            "etf_secondary_short_score",
        ) if c in df.columns
    ]

    if long_cols:
        styler = styler.apply(color_long, subset=long_cols)
    if short_cols:
        styler = styler.apply(color_short, subset=short_cols)

    return styler
"""


def style_etf_scores(df: pd.DataFrame) -> Styler:
    """
    Shade ETF scores:
      - long_score >= 4 => green-ish (stronger if >= 6)
      - short_score >= 4 => red-ish (stronger if >= 6)
      - otherwise => neutral

    Works with flat OR MultiIndex columns.
    """
    def color_long(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v) or v < 4:
                styles.append("")
            elif v >= 6:
                styles.append("background-color: rgba(0, 170, 0, 0.25); color: #006400;")
            else:
                styles.append("background-color: rgba(0, 170, 0, 0.15); color: #006400;")
        return styles

    def color_short(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v) or v < 4:
                styles.append("")
            elif v >= 6:
                styles.append("background-color: rgba(220, 20, 60, 0.25); color: #8B0000;")
            else:
                styles.append("background-color: rgba(220, 20, 60, 0.15); color: #8B0000;")
        return styles

    styler = df.style

    cols = df.columns

    def _find_cols_by_last_level(names: set[str]) -> list:
        if isinstance(cols, pd.MultiIndex):
            return [c for c in cols if c[-1] in names]
        return [c for c in cols if c in names]

    long_names = {"etf_primary_long_score", "etf_secondary_long_score"}
    short_names = {"etf_primary_short_score", "etf_secondary_short_score"}

    long_cols = _find_cols_by_last_level(long_names)
    short_cols = _find_cols_by_last_level(short_names)

    if long_cols:
        styler = styler.apply(color_long, subset=long_cols)
    if short_cols:
        styler = styler.apply(color_short, subset=short_cols)

    return styler


def apply_signal_row_styles(obj):
    """
    Apply subtle row background coloring based on the 'signal' column.

    - Accepts either a DataFrame or a Styler.
    - Handles both flat and MultiIndex columns.
    - Returns a Styler.
    """
    # Handle both DataFrame and Styler inputs
    if isinstance(obj, Styler):
        styler = obj
        df = obj.data
    else:
        df = obj
        styler = df.style

    # ---- resolve the 'signal' column key (flat or MultiIndex) ----
    signal_key = None

    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        for col in cols:
            # look for any column whose *last* level is 'signal'
            if col[-1] == "signal":
                signal_key = col
                break
    else:
        if "signal" in cols:
            signal_key = "signal"

    if signal_key is None:
        # no signal column; nothing to do
        return styler

    def _row_style(row: pd.Series):
        # Series index matches df.columns (string or tuple)
        sig = row.get(signal_key, "none")
        if pd.isna(sig):
            sig = "none"
        sig = str(sig)

        # default transparent
        color = ""

        if sig == "long":
            color = "rgba(0, 128, 0, 0.08)"      # soft green
        elif sig == "short":
            color = "rgba(200, 0, 0, 0.08)"     # soft red
        elif sig == "watch":
            color = "rgba(200, 160, 0, 0.08)"   # soft yellow
        elif sig == "none":
            color = ""  # no shading

        if color:
            return [f"background-color: {color}"] * len(row)
        else:
            return [""] * len(row)

    styler = styler.apply(_row_style, axis=1)
    return styler
