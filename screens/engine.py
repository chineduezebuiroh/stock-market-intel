# screens/engine.py
from __future__ import annotations

import pandas as pd
from typing import Dict, Any


def run_screen(df: pd.DataFrame, screen_cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    TEMPORARY NO-OP SCREEN ENGINE

    Right now we only want snapshots to preserve the raw columns from ingest
    (open, high, low, close, volume, indicators, etc.) without creating any
    per-symbol pivot columns like ('open', 'AAPL').

    So this function deliberately:
      - ignores the YAML screen config
      - does NOT pivot, groupby, stack, unstack, etc.
      - simply returns a shallow copy of the input DataFrame

    Once the architecture is stable, we will reintroduce rule logic here
    in a way that only adds boolean flag columns (e.g. passed_rsi_band),
    without touching the base OHLCV / indicator columns.
    """
    if df is None or df.empty:
        return df

    # Just return a copy so downstream code can mutate safely if needed
    return df.copy()
