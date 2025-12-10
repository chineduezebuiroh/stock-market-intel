from __future__ import annotations

# indicators/composite_spy_qqq_volume_ma_ratio.py
from functools import lru_cache
import numpy as np
import pandas as pd

#from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve

from etl.sources import load_eod, load_130m_from_5m, load_quarterly_from_monthly, load_yearly_from_monthly


@lru_cache(maxsize=16)
def _spy_qqq_vol_ma_for_timeframe(timeframe: str, length: int) -> tuple[pd.Series, pd.Series]:
    """
    Return (spy_vol_ma, qqq_vol_ma) series for the given timeframe.

    Index will be the native index for that timeframe (daily, weekly, intraday_130m, ...).
    """
    if timeframe == "intraday_130m":
        spy_df = load_130m_from_5m("SPY")
        qqq_df = load_130m_from_5m("QQQ")
    elif timeframe == "quarterly":
        spy_df = load_quarterly_from_monthly("SPY", window_bars=length)
        qqq_df = load_quarterly_from_monthly("QQQ", window_bars=length)
    elif timeframe == "yearly":
        spy_df = load_yearly_from_monthly("SPY", window_bars=length)
        qqq_df = load_yearly_from_monthly("QQQ", window_bars=length)
    else:
        # EOD-style for D/W/M, exactly what we've been doing
        spy_df = load_eod("SPY", timeframe=timeframe)
        qqq_df = load_eod("QQQ", timeframe=timeframe)

    # Normalize column names, then compute rolling volume MA
    for df in (spy_df, qqq_df):
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    spy_vol = spy_df["volume"].astype(float)
    qqq_vol = qqq_df["volume"].astype(float)

    # Use the same adaptive window logic as the symbol side
    L = int(length)
    n_spy = len(spy_vol)
    n_qqq = len(qqq_vol)

    # Enough data for SPY/QQQ? they usually have tons, so this is mostly defensive
    window_spy = min(L, n_spy)
    window_qqq = min(L, n_qqq)

    minp_spy = max(3, window_spy // 2)
    minp_qqq = max(3, window_qqq // 2)
    """
    spy_ma = spy_vol.rolling(window=window_spy, min_periods=minp_spy).mean()
    qqq_ma = qqq_vol.rolling(window=window_qqq, min_periods=minp_qqq).mean()
    """
    spy_ma = spy_vol.rolling(window=L, min_periods=minp_spy).mean()
    qqq_ma = qqq_vol.rolling(window=L, min_periods=minp_qqq).mean()
    
    """
    spy_ma = spy_vol.rolling(length, min_periods=length).mean()
    qqq_ma = qqq_vol.rolling(length, min_periods=length).mean()
    """
    return spy_ma, qqq_ma


def indicator_spy_qqq_volume_ma_ratio(
    df: pd.DataFrame,
    symbol_spy: str = "SPY",
    symbol_qqq: str = "QQQ",
    length: int = 26,
    timeframe_name: str = "daily",
    **_,
) -> pd.Series:
    """
    SPY/QQQ Volume MA Ratio (Thinkorswim port).

    For each bar:
        ratio = SMA(volume(symbol), length) / min(SMA(volume(SPY), length),
                                                  SMA(volume(QQQ), length))

    Returns a float Series. Values > 1 indicate the symbol's volume MA
    exceeds the lower of SPY and QQQ volume MAs; < 1 means it's lighter.
    """

    if "volume" not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    # Ensure chronological order
    df_sorted = df.sort_index()
    volume = df_sorted["volume"].astype(float)

    L = int(length)
    """
    if len(volume) < L:
        return pd.Series(index=df.index, dtype="float64")
    """

    n = len(volume)

    # If there are *really* no bars, just return NaN
    if n == 0:
        return pd.Series(index=df.index, dtype="float64")

    # ✅ Adaptive window:
    # - default to L (e.g. 26)
    # - cap at available history
    # - require at least a small minimum to avoid nonsense (e.g. 4 bars)
    window = min(L, n)
    minp = max(4, window // 2)
    
    """
    # SMA of the symbol's own volume
    vol_ma_symbol = volume.rolling(window=L, min_periods=L).mean()
    spy_ma, qqq_ma = _spy_qqq_vol_ma_for_timeframe(timeframe_name, L)
    """

    # SMA of the symbol's own volume
    vol_ma_symbol = volume.rolling(window=window, min_periods=minp).mean()
    # SPY/QQQ MAs with same effective window
    spy_ma, qqq_ma = _spy_qqq_vol_ma_for_timeframe(timeframe_name, window)
    
    
    # Align SPY/QQQ to this symbol’s index without inventing future values.
    spy_aligned = spy_ma.reindex(df_sorted.index)
    qqq_aligned = qqq_ma.reindex(df_sorted.index)

    denom = np.minimum(spy_aligned, qqq_aligned).replace(0, np.nan)

    ratio = vol_ma_symbol / denom
    return ratio.astype("float64")
