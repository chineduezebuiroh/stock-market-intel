from __future__ import annotations

# indicators/composite_macdv.py
import numpy as np
import pandas as pd

from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve


def indicator_macdv(
    df: pd.DataFrame,
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
    atr_length: int = 26,
    macdv_threshold: float = 45.0,
    z: int = 0,
    **_,
) -> pd.Series:
    """
    MACD-V (Thinkorswim port).

    MACD-V = (fastEMA(close) - slowEMA(close)) / ATR(atr_length) * 100
    Signal = EMA(MACD-V, signal_length)

    Encoding (matches TOS scan logic):
        +2  = strong bullish MACD-V thrust
        -2  = strong bearish MACD-V thrust
        +1  = bullish MACD-V bias / improving spread
        -1  = bearish MACD-V bias / worsening spread
         0  = otherwise

    Uses bar offsets like macdv[z], macdv[z+1], macdv[z+2] where z is bars ago.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        return pd.Series(index=df.index, dtype="float64")

    # Ensure chronological order
    df_sorted = df.sort_index()
    price = df_sorted["close"].astype(float)

    # ------------------------------------------------------------------
    # MACD-V core: fast/slow EMAs and ATR-normalized difference
    # ------------------------------------------------------------------
    fast_ema = _ema(price, fast_length)
    slow_ema = _ema(price, slow_length)

    # ATR over atr_length bars
    atr = _atr(df_sorted, atr_length)
    atr_safe = atr.replace(0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        macdv = (fast_ema - slow_ema) / atr_safe * 100.0

    # Signal line: EMA of MACD-V
    signal = _ema(macdv, signal_length)

    # ------------------------------------------------------------------
    # Shifted versions to emulate macdv[z], macdv[z+1], macdv[z+2], etc.
    # In TOS, x[1] = one bar ago, so we use Series.shift(1).
    # ------------------------------------------------------------------
    z_int = int(z)

    macdv_z   = macdv.shift(z_int)
    macdv_z1  = macdv.shift(z_int + 1)
    macdv_z2  = macdv.shift(z_int + 2)

    signal_z  = signal.shift(z_int)
    signal_z1 = signal.shift(z_int + 1)
    signal_z2 = signal.shift(z_int + 2)

    th = float(macdv_threshold)

    # max(macdv[z], macdv[z+1]) and min(...) equivalents
    macdv_max_z_z1 = macdv_z.combine(macdv_z1, np.maximum)
    macdv_max_z1_z2 = macdv_z1.combine(macdv_z2, np.maximum)

    macdv_min_z_z1 = macdv_z.combine(macdv_z1, np.minimum)
    macdv_min_z1_z2 = macdv_z1.combine(macdv_z2, np.minimum)

    # ------------------------------------------------------------------
    # Strong bullish / bearish conditions (value 2 / -2)
    # ------------------------------------------------------------------
    strong_bull_1 = (
        (macdv_max_z_z1 > th)
        & (macdv_z > signal_z)
        & (signal_z > signal_z1)
    )

    strong_bull_2 = (
        (macdv_max_z1_z2 > th)
        & (macdv_z1 > signal_z1)
        & (signal_z1 > signal_z2)
    )

    strong_bull = strong_bull_1 | strong_bull_2

    strong_bear_1 = (
        (macdv_min_z_z1 < -th)
        & (macdv_z < signal_z)
        & (signal_z < signal_z1)
    )

    strong_bear_2 = (
        (macdv_min_z1_z2 < -th)
        & (macdv_z1 < signal_z1)
        & (signal_z1 < signal_z2)
    )

    strong_bear = strong_bear_1 | strong_bear_2

    # ------------------------------------------------------------------
    # Weaker bullish / bearish conditions (value 1 / -1)
    # ------------------------------------------------------------------
    # d = macdv - signal, approximating "spread" behavior
    d_z  = macdv_z - signal_z
    d_z1 = macdv_z1 - signal_z1
    d_z2 = macdv_z2 - signal_z2

    # Bullish:
    #   max(macdv[z], macdv[z+1]) > threshold
    #   OR (macdv[z] > signal[z] > signal[z+1])
    #   OR (d_z > d_z1 > d_z2)  (improving spread)
    bull_1 = macdv_max_z_z1 > th
    bull_2 = (macdv_z > signal_z) & (signal_z > signal_z1)
    bull_3 = (d_z > d_z1) & (d_z1 > d_z2)

    bull = bull_1 | bull_2 | bull_3

    # Bearish:
    #   min(macdv[z], macdv[z+1]) < -threshold
    #   OR (macdv[z] < signal[z] < signal[z+1])
    #   OR (d_z < d_z1 < d_z2) (worsening negative spread)
    bear_1 = macdv_min_z_z1 < -th
    bear_2 = (macdv_z < signal_z) & (signal_z < signal_z1)
    bear_3 = (d_z < d_z1) & (d_z1 < d_z2)

    bear = bear_1 | bear_2 | bear_3

    # ------------------------------------------------------------------
    # Combine with priority:
    #   strong bull (2) > strong bear (-2) > bull (1) > bear (-1) > 0
    # ------------------------------------------------------------------
    scan = pd.Series(0.0, index=df_sorted.index)

    # Apply weaker conditions first...
    scan[bear] = -1.0
    scan[bull] = 1.0
    # ...then overwrite with strong signals where applicable
    scan[strong_bear] = -2.0
    scan[strong_bull] = 2.0

    return scan.reindex(df.index)



def indicator_macdv_guardrail(
    df: pd.DataFrame,
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
    atr_length: int = 26,
    macdv_threshold: float = 45.0,
    z: int = 0,
    **_,
) -> pd.Series:
    """
    MACD-V (Thinkorswim port).

    MACD-V = (fastEMA(close) - slowEMA(close)) / ATR(atr_length) * 100
    Signal = EMA(MACD-V, signal_length)

    Encoding (matches TOS scan logic):
        +1  = strong bullish MACD-V thrust
        -1  = strong bearish MACD-V thrust
         0  = otherwise

    Uses bar offsets like macdv[z], macdv[z+1], macdv[z+2] where z is bars ago.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        return pd.Series(index=df.index, dtype="float64")

    # Ensure chronological order
    df_sorted = df.sort_index()
    price = df_sorted["close"].astype(float)

    # ------------------------------------------------------------------
    # MACD-V core: fast/slow EMAs and ATR-normalized difference
    # ------------------------------------------------------------------
    fast_ema = _ema(price, fast_length)
    slow_ema = _ema(price, slow_length)

    # ATR over atr_length bars
    atr = _atr(df_sorted, atr_length)
    atr_safe = atr.replace(0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        macdv = (fast_ema - slow_ema) / atr_safe * 100.0

    # Signal line: EMA of MACD-V
    signal = _ema(macdv, signal_length)

    # ------------------------------------------------------------------
    # Shifted versions to emulate macdv[z], macdv[z+1], macdv[z+2], etc.
    # In TOS, x[1] = one bar ago, so we use Series.shift(1).
    # ------------------------------------------------------------------
    z_int = int(z)

    macdv_z   = macdv.shift(z_int)
    macdv_z1  = macdv.shift(z_int + 1)

    signal_z  = signal.shift(z_int)
    signal_z1 = signal.shift(z_int + 1)

    th = float(macdv_threshold)

    # max(macdv[z], macdv[z+1]) and min(...) equivalents
    macdv_max_z_z1 = macdv_z.combine(macdv_z1, np.maximum)

    macdv_min_z_z1 = macdv_z.combine(macdv_z1, np.minimum)

    # ------------------------------------------------------------------
    # Strong bullish / bearish conditions (value 1 / -1)
    # ------------------------------------------------------------------
    strong_bull = (
        (macdv_max_z_z1 > th)
        & (macdv_z > signal_z)
        & (signal_z > signal_z1)
    )

    strong_bear = (
        (macdv_min_z_z1 < -th)
        & (macdv_z < signal_z)
        & (signal_z < signal_z1)
    )

    # ------------------------------------------------------------------
    # Combine with priority: strong bull (1) > strong bear (-1) > 0
    # ------------------------------------------------------------------
    scan = pd.Series(0.0, index=df_sorted.index)

    # ...strong signals where applicable
    scan[strong_bear] = -1.0
    scan[strong_bull] = 1.0

    return scan.reindex(df.index)
