from __future__ import annotations

# indicators/composite_spy_qqq_volume_ma_ratio.py
import numpy as np
import pandas as pd

from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve


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
