from __future__ import annotations

# app/data_access.py

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from core.paths import DATA
from core import storage


def load_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Load parquet if it exists (via storage backend), else None."""
    if not storage.exists(path):
        return None
    try:
        return storage.load_parquet(path)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return None


def load_combo_safe(combo_name: str) -> pd.DataFrame:
    """
    Load data/combo_<combo_name>.parquet, or return empty DataFrame.
    Ensures core columns exist.
    """
    path = DATA / f"combo_{combo_name}.parquet"
    df = load_parquet_safe(path)
    if df is None:
        return pd.DataFrame()

    for col in ["symbol", "signal", "signal_side", "mtf_long_score", "mtf_short_score"]:
        if col not in df.columns:
            df[col] = np.nan

    # Backfill signal_side if missing
    if "signal_side" in df.columns and "signal" in df.columns:
        df["signal_side"] = df["signal_side"].fillna(df["signal"])

    return df
