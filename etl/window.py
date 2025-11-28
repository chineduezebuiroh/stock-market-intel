from pathlib import Path
import pandas as pd

def parquet_path(root: Path, timeframe: str, ticker: str) -> Path:
    return root / f"timeframe={timeframe}" / f"ticker={ticker}" / "data.parquet"


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the index is a tz-naive DatetimeIndex (or leave empty frames alone).
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # If not a DatetimeIndex, try to convert
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Drop timezone if present (tz-naive)
    if getattr(df.index, "tz", None) is not None:
        #df.index = df.index.tz_localize(None)
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    return df


"""
def update_fixed_window(df_new: pd.DataFrame, existing: pd.DataFrame, window_bars: int) -> pd.DataFrame:
    df = pd.concat([existing, df_new]).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if len(df) > window_bars:
        df = df.iloc[-window_bars:]
    return df
"""


def update_fixed_window(df_new: pd.DataFrame, existing: pd.DataFrame, window_bars: int) -> pd.DataFrame:
    """
    Concatenate existing + new data, sort by index, drop duplicates,
    and keep only the last `window_bars` rows.
    """
    existing = _normalize_index(existing)
    df_new = _normalize_index(df_new)

    if existing is None or existing.empty:
        df = df_new
    elif df_new is None or df_new.empty:
        df = existing
    else:
        df = pd.concat([existing, df_new])
        # Drop duplicate index entries, keep last occurrence
        df = df[~df.index.duplicated(keep="last")]

    if df is None or df.empty:
        return df

    df = df.sort_index()

    if window_bars is not None:
        df = df.iloc[-window_bars:]

    return df
