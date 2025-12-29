from __future__ import annotations

# etl/futures_partial.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd

from core.guard import now_ny, in_futures_session

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

    out.index = pd.to_datetime(out.index)  # ok even if empty

    # Align by column name (safer than .values)
    out.loc[ts, bar.index] = bar

    return out.sort_index()


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

"""
def patch_partial_1h_from_5m(
    *,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    now: Optional[datetime] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    
    base = df_1h.copy() if df_1h is not None and not df_1h.empty else pd.DataFrame()

    if df_5m is None or df_5m.empty:
        return base

    # Normalize indexes (NY tz-naive convention)
    if not base.empty:
        base.index = pd.to_datetime(base.index)

    df_5m = df_5m.copy()
    idx5 = pd.to_datetime(df_5m.index)
    # robust tz stripping
    if getattr(idx5, "tz", None) is not None:
        idx5 = idx5.tz_convert("America/New_York").tz_localize(None)
    df_5m.index = idx5
    df_5m = df_5m.sort_index()

    hour_start = pd.to_datetime(current_hour_start(now))
    hour_end = hour_start + pd.Timedelta(hours=1)

    # If a real 1h row already exists at hour_start (with nonzero vol), don't stomp it.
    if not base.empty and hour_start in pd.to_datetime(base.index):
        try:
            v = base.loc[hour_start, "volume"] if "volume" in base.columns else None
            if v is not None and float(v) > 0:
                return base
        except Exception:
            pass

    # 1) If we have 5m prints inside the current hour, aggregate them
    chunk = df_5m.loc[(df_5m.index >= hour_start) & (df_5m.index < hour_end)]
    if not chunk.empty:
        bar = _agg_ohlcv(chunk.sort_index())
        # ensure adj_close exists
        if "adj_close" not in bar.index:
            bar["adj_close"] = bar.get("close", float("nan"))
        return _upsert_bar(base, hour_start, bar)

    # 2) Otherwise create carry-forward stub
    last_close = None

    if not base.empty and "close" in base.columns:
        s = base.sort_index()["close"].dropna()
        if not s.empty:
            last_close = float(s.iloc[-1])

    if last_close is None and "close" in df_5m.columns:
        prev_5m = df_5m.loc[df_5m.index < hour_start]
        s = prev_5m["close"].dropna()
        if not s.empty:
            last_close = float(s.iloc[-1])

    if last_close is None:
        return base

    sym = f"{symbol} " if symbol else ""
    print(
        f"[STUB][1H] {sym}@ {hour_start} (vol=0, close={last_close})",
        flush=True,
    )

    stub = pd.Series(
        {
            "open": last_close,
            "high": last_close,
            "low": last_close,
            "close": last_close,
            "adj_close": last_close,
            "volume": 0.0,
        },
        dtype="float64",
    )

    return _upsert_bar(base, hour_start, stub)
"""


def patch_partial_1h_from_5m(
    *,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    now: Optional[datetime] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:

    base = df_1h.copy() if df_1h is not None and not df_1h.empty else pd.DataFrame()

    if df_5m is None or df_5m.empty:
        return base

    if not base.empty:
        base.index = pd.to_datetime(base.index)

    df_5m = df_5m.copy()
    idx5 = pd.to_datetime(df_5m.index)
    if getattr(idx5, "tz", None) is not None:
        idx5 = idx5.tz_convert("America/New_York").tz_localize(None)
    df_5m.index = idx5
    df_5m = df_5m.sort_index()

    hour_start = pd.to_datetime(current_hour_start(now))
    hour_end = hour_start + pd.Timedelta(hours=1)

    # ✅ CHANGE 2: never stub outside futures session
    # if not in_futures_session(now_ny() if now is None else now):
    #     return base
    now_eff = now_ny() if now is None else now
    if not in_futures_session(now_eff):
        return base

    existing_row_exists = (not base.empty and hour_start in base.index)

    # 1) If we have 5m prints inside the current hour, aggregate them (OK to overwrite)
    chunk = df_5m.loc[(df_5m.index >= hour_start) & (df_5m.index < hour_end)]
    if not chunk.empty:
        bar = _agg_ohlcv(chunk)
        if "adj_close" not in bar.index:
            bar["adj_close"] = bar.get("close", float("nan"))
        return _upsert_bar(base, hour_start, bar)

    # ✅ CHANGE 1: if bar already exists but no 5m prints, do NOT stub over it
    if existing_row_exists:
        return base

    # 2) Otherwise create carry-forward stub
    last_close = None

    if not base.empty and "close" in base.columns:
        s = base.sort_index()["close"].dropna()
        if not s.empty:
            last_close = float(s.iloc[-1])

    if last_close is None and "close" in df_5m.columns:
        prev_5m = df_5m.loc[df_5m.index < hour_start]
        s = prev_5m["close"].dropna()
        if not s.empty:
            last_close = float(s.iloc[-1])

    if last_close is None:
        return base

    sym = f"{symbol} " if symbol else ""
    print(f"[STUB][1H] {sym}@ {hour_start} (vol=0, close={last_close})", flush=True)

    stub = pd.Series(
        {
            "open": last_close,
            "high": last_close,
            "low": last_close,
            "close": last_close,
            "adj_close": last_close,
            "volume": 0.0,
        },
        dtype="float64",
    )

    return _upsert_bar(base, hour_start, stub)


def patch_partial_4h_from_5m(
    *,
    df_4h: pd.DataFrame,
    df_5m: pd.DataFrame,
    now: Optional[datetime] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    
    base = df_4h.copy() if df_4h is not None and not df_4h.empty else pd.DataFrame()

    if df_5m is None or df_5m.empty:
        return base

    # Normalize indexes (NY tz-naive convention)
    if not base.empty:
        base.index = pd.to_datetime(base.index)

    df_5m = df_5m.copy()
    idx5 = pd.to_datetime(df_5m.index)
    # robust tz stripping
    if getattr(idx5, "tz", None) is not None:
        idx5 = idx5.tz_convert("America/New_York").tz_localize(None)
    df_5m.index = idx5
    df_5m = df_5m.sort_index()

    bucket_start = pd.to_datetime(current_4h_bucket_start_5pm_anchor(now))
    bucket_end = bucket_start + pd.Timedelta(hours=4)

    # ✅ CHANGE 2: never stub outside futures session
    # (place in a shared module; this is just the call site)
    # if not in_futures_session(now_ny() if now is None else now):
    #     return base
    now_eff = now_ny() if now is None else now
    if not in_futures_session(now_eff):
        return base

    existing_row_exists = (not base.empty and bucket_start in base.index)

    # 1) If we have 5m prints inside the current bucket, aggregate them (OK to overwrite)
    chunk = df_5m.loc[(df_5m.index >= bucket_start) & (df_5m.index < bucket_end)]
    if not chunk.empty:
        bar = _agg_ohlcv(chunk)
        if "adj_close" not in bar.index:
            bar["adj_close"] = bar.get("close", float("nan"))
        return _upsert_bar(base, bucket_start, bar)

    # ✅ CHANGE 1: if bar already exists but no 5m prints, do NOT stub over it
    if existing_row_exists:
        return base

    # 2) Otherwise create carry-forward stub
    last_close = None

    if not base.empty and "close" in base.columns:
        s = base.sort_index()["close"].dropna()
        if not s.empty:
            last_close = float(s.iloc[-1])

    if last_close is None and "close" in df_5m.columns:
        prev_5m = df_5m.loc[df_5m.index < bucket_start]
        s = prev_5m["close"].dropna()
        if not s.empty:
            last_close = float(s.iloc[-1])

    if last_close is None:
        return base

    sym = f"{symbol} " if symbol else ""
    print(f"[STUB][4H] {sym}@ {bucket_start} (vol=0, close={last_close})", flush=True)

    stub = pd.Series(
        {
            "open": last_close,
            "high": last_close,
            "low": last_close,
            "close": last_close,
            "adj_close": last_close,
            "volume": 0.0,
        },
        dtype="float64",
    )

    return _upsert_bar(base, bucket_start, stub)
    
