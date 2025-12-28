from __future__ import annotations

# app/combo_view.py

from typing import Iterable, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from app.field_catalog import TIME_COL_CANDIDATES


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
        tmp = df.copy()
        tmp["mtf_abs_max"] = tmp[["mtf_long_score", "mtf_short_score"]].abs().max(axis=1)
        tmp = tmp.sort_values("mtf_abs_max", ascending=False, na_position="last")
        tmp = tmp.drop(columns=["mtf_abs_max"])
        return tmp

    return df


def add_score_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "mtf_long_score" not in df.columns or "mtf_short_score" not in df.columns:
        return df

    out = df.copy()

    def _fmt(row):
        ls = row.get("mtf_long_score", np.nan)
        ss = row.get("mtf_short_score", np.nan)
        try:
            if pd.isna(ls) and pd.isna(ss):
                return ""
            return f"{float(ls):.1f} / {float(ss):.1f}"
        except Exception:
            return ""

    out["score_summary"] = out.apply(_fmt, axis=1)
    return out


def add_mtf_grouped_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Convert lower_/middle_/upper_ prefixes into MultiIndex top-level headers."""
    if df is None or df.empty:
        return df

    new_cols = []
    for c in df.columns:
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


def _time_cols_present(df: pd.DataFrame) -> list[str]:
    return [c for c in TIME_COL_CANDIDATES if c in df.columns]


def select_display_cols(
    df: pd.DataFrame,
    base_cols: list[str],
    *,
    include_time_cols: bool = True,
) -> pd.DataFrame:
    """Selects a preferred ordered subset of columns (if present)."""
    if df.empty:
        return df
    cols = base_cols.copy()

    if include_time_cols:
        tcols = _time_cols_present(df)
        # Insert time cols right after signal if present
        if "signal" in cols:
            i = cols.index("signal") + 1
            cols = cols[:i] + tcols + cols[i:]
        else:
            cols = tcols + cols

    out_cols = [c for c in cols if c in df.columns]
    return df[out_cols] if out_cols else df


# -----------------------------
# Collapsible columns UI
# -----------------------------
def filter_combo_columns_ui(
    df_view_flat: pd.DataFrame,
    *,
    key_prefix: str,
) -> pd.DataFrame:
    """
    Show everything by default, but let user hide groups:
      META, LOWER, MIDDLE, UPPER

    Works on a FLAT df (string columns), before add_mtf_grouped_headers().
    """
    if df_view_flat is None or df_view_flat.empty:
        return df_view_flat

    # Determine which groups exist
    groups = {
        "META": [c for c in df_view_flat.columns if not (str(c).startswith(("lower_", "middle_", "upper_")))],
        "LOWER": [c for c in df_view_flat.columns if str(c).startswith("lower_")],
        "MIDDLE": [c for c in df_view_flat.columns if str(c).startswith("middle_")],
        "UPPER": [c for c in df_view_flat.columns if str(c).startswith("upper_")],
    }
    groups = {k: v for k, v in groups.items() if v}

    with st.expander("Columns (collapse groups)", expanded=False):
        cols = st.columns(len(groups))
        show = {}
        for i, (g, _) in enumerate(groups.items()):
            show[g] = cols[i].checkbox(f"Show {g}", value=True, key=f"{key_prefix}_show_{g}")

    keep_cols = []
    for g, cols_list in groups.items():
        if show.get(g, True):
            keep_cols.extend(cols_list)

    # Preserve original order
    keep_set = set(keep_cols)
    ordered = [c for c in df_view_flat.columns if c in keep_set]
    return df_view_flat[ordered]
