from __future__ import annotations

# etl/futures_partial.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd

from core.guard import now_ny

from etl.session import ensure_ny_index, NY


_OHLCV = ["open", "high", "low", "close", "adj_close", "volume"]


def _agg_ohlcv(df: pd.DataFrame) -> pd.Series:
    # assumes df sorted by index
    s = {
        "open": df["open"].iloc[0],
        "high": df["high"].max(),
        "low": df["low"].min(),
        "close": df["close"].iloc[-1],
        "volume": float(df["volume"].sum()),
    }
    if "adj_close" in df.columns:
        s["adj_close"] = df["adj_close"].iloc[-1]
    else:
        s["adj_close"] = s["close"]
    return pd.Series(s)


def _upsert_bar(df: pd.DataFrame, ts: pd.Timestamp, bar: pd.Series) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(ts)

    # Ensure same index type/naivety as out
    out_idx = pd.to_datetime(out.index)
    out.index = out_idx

    # If row exists: overwrite; else append
    out.loc[ts, bar.index] = bar.values
    out = out.sort_index()
    return out


def _try_load_recent_intraday(load_intraday_fn, symbol: str, session: str) -> pd.DataFrame:
    """
    Try 5m first; if unavailable, fall back to 15m.
    Keep this narrow (few days) – just for partial bars.
    """
    for interval, period in [("5m", "7d"), ("15m", "30d")]:
        df = load_intraday_fn(symbol, interval=interval, period=period, session=session)
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()


def current_hour_start(now: Optional[datetime] = None) -> pd.Timestamp:
    if now is None:
        now = now_ny()  # tz-aware NY
    # Convert to NY, floor to hour, then drop tz to match your parquet convention
    n = pd.Timestamp(now.astimezone(NY)).replace(minute=0, second=0, microsecond=0)
    return n.tz_localize(None)


def current_4h_bucket_start_5pm_anchor(now: Optional[datetime] = None) -> pd.Timestamp:
    """
    4h buckets anchored so that 17:00 ET is a boundary.
    Bucket starts: 17, 21, 1, 5, 9, 13.
    """
    if now is None:
        now = now_ny()
    
    # Work in naive NY timestamps (matches your parquet convention)
    n = pd.Timestamp(now.astimezone(NY)).tz_localize(None)

    # Shift so 17:00 becomes 00:00, then floor to 4h, then shift back
    shifted = n - pd.Timedelta(hours=17)
    bucket = shifted.floor("4h") + pd.Timedelta(hours=17)
    return bucket


def patch_partial_1h_from_5m(
    *,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    if df_1h is None or df_1h.empty:
        base = pd.DataFrame()
    else:
        base = df_1h.copy()

    if df_5m is None or df_5m.empty:
        return base

    base.index = pd.to_datetime(base.index)
    df_5m = df_5m.copy()
    df_5m.index = pd.to_datetime(df_5m.index)

    hour_start = current_hour_start(now)
    hour_end = hour_start + pd.Timedelta(hours=1)

    chunk = df_5m.loc[(df_5m.index >= hour_start) & (df_5m.index < hour_end)].sort_index()
    if chunk.empty:
        return base

    bar = _agg_ohlcv(chunk)
    return _upsert_bar(base, hour_start, bar)


def patch_partial_4h_from_5m(
    *,
    df_4h: pd.DataFrame,
    df_5m: pd.DataFrame,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    if df_4h is None or df_4h.empty:
        base = pd.DataFrame()
    else:
        base = df_4h.copy()

    if df_5m is None or df_5m.empty:
        return base

    base.index = pd.to_datetime(base.index)
    df_5m = df_5m.copy()
    df_5m.index = pd.to_datetime(df_5m.index)

    bucket_start = current_4h_bucket_start_5pm_anchor(now)
    bucket_end = bucket_start + pd.Timedelta(hours=4)

    chunk = df_5m.loc[(df_5m.index >= bucket_start) & (df_5m.index < bucket_end)].sort_index()
    if chunk.empty:
        return base

    bar = _agg_ohlcv(chunk)
    return _upsert_bar(base, bucket_start, bar)
