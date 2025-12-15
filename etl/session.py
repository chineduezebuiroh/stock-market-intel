from __future__ import annotations

# etl/session.py

from datetime import time
import pandas as pd
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# CME Globex "trade date" rolls at ~6pm NY
SESSION_ROLLOVER = time(18, 0)  # 6:00pm


def ensure_ny_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Ensure idx is tz-aware and converted to America/New_York.
    """
    if idx.tz is None:
        # Choose one: treat naive timestamps as NY-local timestamps
        return idx.tz_localize(NY)
    return idx.tz_convert(NY)


def trade_date_from_ts(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Map a timestamp to its session "trade date" using the 6pm NY rollover rule.

    If local time >= 18:00, trade_date = next calendar day
    else trade_date = same calendar day.

    Returns a normalized date-like Timestamp at midnight (NY date).
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY)
    else:
        ts = ts.tz_convert(NY)

    d = ts.normalize()  # midnight of local calendar day
    if ts.time() >= SESSION_ROLLOVER:
        d = d + pd.Timedelta(days=1)

    # Optional hardening: if your feed ever has Saturday prints, map them forward.
    # Saturday = 5 (Mon=0)
    if d.weekday() == 5:
        d = d + pd.Timedelta(days=2)

    return d


def add_trade_date(df: pd.DataFrame, ts_col: str | None = None) -> pd.DataFrame:
    """
    Adds a 'trade_date' column (Timestamp at midnight, NY date).
    Uses df.index if ts_col is None, else uses df[ts_col].
    """
    out = df.copy()

    if ts_col is None:
        idx = ensure_ny_index(out.index)
        out["trade_date"] = pd.Index(idx).map(trade_date_from_ts)
    else:
        s = pd.to_datetime(out[ts_col])
        # make tz-aware in NY
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize(NY)
        else:
            s = s.dt.tz_convert(NY)
        out["trade_date"] = s.map(trade_date_from_ts)

    return out


def drop_maintenance_break_1h(df: pd.DataFrame) -> pd.DataFrame:
    """
    CME Globex maintenance break is ~17:00–18:00 NY.
    With bar-start indexing, the 17:00 bar represents the maintenance hour.
    Drop it to avoid contaminating aggregates.
    """
    if df.empty:
        return df

    out = df.copy()
    idx = ensure_ny_index(pd.to_datetime(out.index))
    out.index = idx

    # Drop any bar that starts in the 17:00 NY hour
    mask = idx.hour != 17
    return out.loc[mask]
