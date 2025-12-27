from __future__ import annotations

# indicators/composite_movingavg_trend_cloud.py
import numpy as np
import pandas as pd

from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve


def _movingavg_trend_cloud_core(
    df: pd.DataFrame,
    *,
    fast_length: int = 11,
    displace: int = 0,
) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series, pd.Series]:
    """
    Shared foundation:
      - validates 'close'
      - sorts chronologically
      - applies displace (TOS-style: price[-displace] => shift(-displace))
      - computes: fast_ema, fast_sma, fast_wilders

    Returns (sorted_index, fast_ema, fast_sma, fast_wilders).
    If invalid/insufficient history: returns empty float series aligned to sorted index.
    """
    if df is None or df.empty or "close" not in df.columns:
        idx = df.index if df is not None else pd.DatetimeIndex([])
        empty = pd.Series(index=idx, dtype="float64")
        return idx, empty, empty, empty

    df_sorted = df.sort_index()
    close = df_sorted["close"].astype(float)

    price = close.shift(-displace) if displace != 0 else close

    fl = int(fast_length)
    if len(price) < fl:
        empty = pd.Series(index=df_sorted.index, dtype="float64")
        return df_sorted.index, empty, empty, empty

    fast_ema = _ema(price, fl)
    fast_sma = _sma(price, fl)
    fast_wilders = _wema(price, fl)

    return df_sorted.index, fast_ema, fast_sma, fast_wilders


def indicator_movingavg_trend_bullish(
    df: pd.DataFrame,
    fast_length: int = 11,
    displace: int = 0,
    **_,
) -> pd.Series:
    """
    Encoding:
        2 = BullishTrend
        1 = BullishTurn (no trend)
        0 = otherwise
    """
    idx_sorted, fast_ema, fast_sma, fast_wilders = _movingavg_trend_cloud_core(
        df, fast_length=fast_length, displace=displace
    )

    # If we returned empties, just return zeros on df.index
    if fast_ema.empty:
        #return pd.Series(0.0, index=df.index, dtype="float64")
        return pd.Series(index=df.index, dtype="float64")

    bullish_trend = (fast_ema > fast_wilders) & (fast_sma > fast_wilders)
    bullish_turn = fast_ema > fast_sma

    scan = pd.Series(0.0, index=idx_sorted)
    scan[bullish_turn] = 1.0
    scan[bullish_trend] = 2.0

    return scan.reindex(df.index)


def indicator_movingavg_trend_bearish(
    df: pd.DataFrame,
    fast_length: int = 11,
    displace: int = 0,
    **_,
) -> pd.Series:
    """
    Encoding:
       -2 = BearishTrend
       -1 = BearishTurn (no trend)
        0 = otherwise
    """
    idx_sorted, fast_ema, fast_sma, fast_wilders = _movingavg_trend_cloud_core(
        df, fast_length=fast_length, displace=displace
    )

    if fast_ema.empty:
        #return pd.Series(0.0, index=df.index, dtype="float64")
        return pd.Series(index=df.index, dtype="float64")

    bearish_trend = (fast_ema < fast_wilders) & (fast_sma < fast_wilders)
    bearish_turn = fast_ema < fast_sma

    scan = pd.Series(0.0, index=idx_sorted)
    scan[bearish_turn] = -1.0
    scan[bearish_trend] = -2.0

    return scan.reindex(df.index)
