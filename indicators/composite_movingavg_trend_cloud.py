from __future__ import annotations

# indicators/composite_movingavg_trend_cloud.py
import numpy as np
import pandas as pd

from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve


def indicator_movingavg_trend_cloud(
    df: pd.DataFrame,
    fast_length: int = 11,
    displace: int = 0,
    **_,
) -> pd.Series:
    """
    MovingAvg_Trend_Cloud (Thinkorswim port).

    Uses fast EMA / SMA / Wilder's MA of (optionally displaced) close:

      - BullishTrend  = fast_EMA > fast_Wilders and fast_SMA > fast_Wilders
      - BearishTrend  = fast_EMA < fast_Wilders and fast_SMA < fast_Wilders
      - BullishTurn   = fast_EMA > fast_SMA
      - BearishTurn   = fast_EMA < fast_SMA

    Encoding (matches TOS scan priority):
        2   = BullishTrend
       -2   = BearishTrend
        1   = BullishTurn (no trend)
       -1   = BearishTurn (no trend)
        0   = otherwise
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    # Ensure chronological order
    df_sorted = df.sort_index()
    close = df_sorted["close"].astype(float)

    # Handle displace similar to price[-displace] in TOS:
    # positive displace => peek forward; negative => lag.
    if displace != 0:
        price = close.shift(-displace)
    else:
        price = close

    # Need at least fast_length bars
    if len(price) < fast_length:
        return pd.Series(index=df.index, dtype="float64")

    # Fast EMA / SMA / Wilder's MA (we already have _ema and _wema helpers)
    fast_ema = _ema(price, fast_length)
    fast_sma = _sma(price, fast_length)
    fast_wilders = _wema(price, fast_length)

    bullish_trend = (fast_ema > fast_wilders) & (fast_sma > fast_wilders)
    bearish_trend = (fast_ema < fast_wilders) & (fast_sma < fast_wilders)

    bullish_turn = fast_ema > fast_sma
    bearish_turn = fast_ema < fast_sma

    # Start from 0, then apply conditions in order of *increasing* priority,
    # so that later assignments (trend) override earlier ones (turn).
    scan = pd.Series(0.0, index=df_sorted.index)

    scan[bearish_turn] = -1.0
    scan[bullish_turn] = 1.0
    scan[bearish_trend] = -2.0
    scan[bullish_trend] = 2.0

    # Reindex to original df index (usually already sorted)
    return scan.reindex(df.index)
  
