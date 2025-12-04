from __future__ import annotations

# indicators/composite_spy_qqq_volume_ma_ratio.py
from functools import lru_cache
import numpy as np
import pandas as pd

from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve

from etl.sources import load_eod, load_130m_from_5m  # adjust import paths if needed


@lru_cache(maxsize=16)
def _spy_qqq_vol_ma_for_timeframe(timeframe: str, length: int) -> tuple[pd.Series, pd.Series]:
    """
    Return (spy_vol_ma, qqq_vol_ma) series for the given timeframe.

    Index will be the native index for that timeframe (daily, weekly, intraday_130m, ...).
    """
    if timeframe == "intraday_130m":
        spy_df = load_130m_from_5m("SPY")
        qqq_df = load_130m_from_5m("QQQ")
    else:
        # EOD-style for D/W/M, exactly what we've been doing
        spy_df = load_eod("SPY")
        qqq_df = load_eod("QQQ")

    # Normalize column names, then compute rolling volume MA
    for df in (spy_df, qqq_df):
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    spy_vol = spy_df["volume"].astype(float)
    qqq_vol = qqq_df["volume"].astype(float)

    spy_ma = spy_vol.rolling(length, min_periods=length).mean()
    qqq_ma = qqq_vol.rolling(length, min_periods=length).mean()

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
    if len(volume) < L:
        return pd.Series(index=df.index, dtype="float64")

    # SMA of the symbol's own volume
    vol_ma_symbol = volume.rolling(window=L, min_periods=L).mean()

    """
    # Load benchmark volume MAs (SPY & QQQ) for this timeframe
    spy_ma_full = _load_benchmark_vol_ma(timeframe_name, symbol_spy, L)
    qqq_ma_full = _load_benchmark_vol_ma(timeframe_name, symbol_qqq, L)
    
    # Align benchmarks to this symbol's index
    spy_ma = spy_ma_full.reindex(df_sorted.index)
    qqq_ma = qqq_ma_full.reindex(df_sorted.index)

    # Take the elementwise minimum of SPY/QQQ volume MA
    bench_min = pd.concat([spy_ma, qqq_ma], axis=1).min(axis=1)

    # Avoid divide-by-zero / negative nonsense
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = vol_ma_symbol / bench_min.replace(0, np.nan)

    ratio = ratio.astype(float)

    # Reindex back to original df index (usually already sorted)
    return ratio.reindex(df.index)
    """

    spy_ma, qqq_ma = _spy_qqq_vol_ma_for_timeframe(timeframe, length)
    
    # Align SPY/QQQ to this symbol’s index without inventing future values.
    spy_aligned = spy_ma.reindex(df_sorted.index)
    qqq_aligned = qqq_ma.reindex(df_sorted.index)

    denom = np.minimum(spy_aligned, qqq_aligned).replace(0, np.nan)

    ratio = vol_ma_symbol / denom
    return ratio.astype("float64")
