from __future__ import annotations

# etl/futures_resample.py

import pandas as pd

from etl.session import add_trade_date, ensure_ny_index, drop_maintenance_break_1h

from core.paths import DATA
from core import storage


_OHLCV = ("open", "high", "low", "close", "adj_close", "volume")


def _agg_ohlcv(g: pd.DataFrame) -> pd.Series:
    close = g["close"].iloc[-1]
    return pd.Series(
        {
            "open": g["open"].iloc[0],
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": close,
            "adj_close": close,          # ✅ add this
            "volume": g["volume"].sum(),
        }
    )



def resample_1h_to_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Build futures daily bars from 1h bars using session trade_date.
    Index returned: tz-naive dates at midnight (trade_date), named 'Date'.
    """
    if df_1h.empty:
        return df_1h

    df = df_1h.copy()
    df.index = ensure_ny_index(pd.to_datetime(df.index))

    # drop maintenance break hour
    df = drop_maintenance_break_1h(df)

    # add trade_date and group
    df = add_trade_date(df)

    daily = df.groupby("trade_date", sort=True, as_index=True).apply(_agg_ohlcv)

    daily.index.name = "Date"
    # match your existing convention: tz-naive midnight timestamps
    daily.index = pd.to_datetime(daily.index).tz_localize(None)

    return daily


def resample_daily_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly bars derived from daily bars. Weeks are ISO weeks of trade_date.
    """
    if df_daily.empty:
        return df_daily

    d = df_daily.copy()
    idx = pd.to_datetime(d.index)

    # idx is tz-naive midnight trade_date; treat as date
    trade_dates = pd.to_datetime(idx.date)
    iso = trade_dates.isocalendar()
    wk_key = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    tmp = d.copy()
    tmp["_wk"] = wk_key.values

    weekly = tmp.groupby("_wk", sort=True, as_index=True).apply(
        lambda g: pd.Series(
            {
                "open": g["open"].iloc[0],
                "high": g["high"].max(),
                "low": g["low"].min(),
                "close": g["close"].iloc[-1],
                "volume": g["volume"].sum(),
            }
        )
    )

    # choose a left-index that is stable: first trade_date in that ISO week
    first_dates = (
        pd.Series(trade_dates.values, index=wk_key.values)
        .groupby(level=0)
        .min()
        .sort_index()
    )
    weekly.index = pd.to_datetime(first_dates.values)
    weekly.index.name = "Date"
    return weekly


def resample_daily_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly bars derived from daily bars using trade_date month (YYYY-MM).
    """
    if df_daily.empty:
        return df_daily

    d = df_daily.copy()
    idx = pd.to_datetime(d.index)
    month_key = idx.to_period("M").astype(str)

    tmp = d.copy()
    tmp["_m"] = month_key

    monthly = tmp.groupby("_m", sort=True, as_index=True).apply(
        lambda g: pd.Series(
            {
                "open": g["open"].iloc[0],
                "high": g["high"].max(),
                "low": g["low"].min(),
                "close": g["close"].iloc[-1],
                "volume": g["volume"].sum(),
            }
        )
    )

    # left-index = first trade_date of month
    first_dates = (
        pd.Series(idx.date, index=month_key)
        .groupby(level=0)
        .min()
        .sort_index()
    )
    monthly.index = pd.to_datetime(first_dates.values)
    monthly.index.name = "Date"
    return monthly


def load_futures_eod_from_1h(
    symbol: str,
    timeframe: str,  # "daily" | "weekly" | "monthly"
    window_bars: int = 300,
) -> pd.DataFrame:
    """
    Build futures daily/weekly/monthly bars from canonical 1h bars.
    Returns only the last `window_bars` rows (like other loaders).
    """
    p1h = DATA / "bars" / "futures_intraday_1h" / f"{symbol}.parquet"
    if not storage.exists(p1h):
        return pd.DataFrame()

    df1h = storage.load_parquet(p1h)
    if df1h is None or df1h.empty:
        return pd.DataFrame()

    daily = resample_1h_to_daily(df1h)

    if timeframe == "daily":
        out = daily
    elif timeframe == "weekly":
        out = resample_daily_to_weekly(daily)
    elif timeframe == "monthly":
        out = resample_daily_to_monthly(daily)
    else:
        raise ValueError(f"Unsupported timeframe for futures EOD-from-1h: {timeframe}")

    if out is None or out.empty:
        return pd.DataFrame()

    return out.tail(window_bars)
