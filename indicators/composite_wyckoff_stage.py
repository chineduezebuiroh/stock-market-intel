from __future__ import annotations

# indicators/composite_wyckoff_stage.py
import numpy as np
import pandas as pd

from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve


def indicator_wyckoff_stage(
    df: pd.DataFrame,
    fast_len: int,
    int_len: int,
    slow_len: int,
    vma_length_source: str = "fast",
    **_,
) -> pd.Series:
    """
    Wyckoff-style stage analysis.

    Numeric encoding:
      +2  = Stage: Accel (close > VMA and bullish VWMA stack)
      -2  = Stage: Decel (close < VMA and bearish VWMA stack)
      +1  = VMA rising (VMA[t] > VMA[t-1]) but not accel/decel
      -1  = VMA falling (VMA[t] < VMA[t-1]) but not accel/decel
       0  = neutral / distribution / no clear stage

    This function does NOT store the VMA or VWMAs as separate columns;
    they are internal to the computation.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    # Ensure chronological order (oldest -> newest), TOS-style.
    df_sorted = df.sort_index()

    close = df_sorted["close"].astype(float)
    volume = df_sorted["volume"].astype(float)

    # Choose the length used for the adaptive VMA
    vma_length_source = (vma_length_source or "fast").lower()
    if vma_length_source.startswith("fast"):
        length = fast_len
    elif vma_length_source.startswith("int"):
        length = int_len
    elif vma_length_source.startswith("slow"):
        length = slow_len
    else:
        length = fast_len  # fallback

    # Need enough history for the longest window
    min_len = max(fast_len, int_len, slow_len, length)
    if len(df_sorted) < min_len:
        return pd.Series(index=df.index, dtype="float64")

    # ------------------------------------------------------------------
    # Adaptive coefficient (TOS logic port)
    # ------------------------------------------------------------------
    diff = close.diff()

    tmp1 = diff.clip(lower=0)            # positive moves
    tmp2 = (-diff).clip(lower=0)         # negative moves

    d2 = tmp1.rolling(length).sum()
    d4 = tmp2.rolling(length).sum()

    denom = d2 + d4
    ad3 = pd.Series(index=close.index, dtype="float64")

    nonzero = denom != 0
    ad3.loc[nonzero] = (d2[nonzero] - d4[nonzero]) / denom[nonzero] * 100.0
    ad3.loc[~nonzero] = 0.0

    coeff = (2.0 / (length + 1.0)) * np.abs(ad3) / 100.0

    # ------------------------------------------------------------------
    # Recursive VMA:
    #   VMA[t] = coeff[t] * price[t] + (1 - coeff[t]) * VMA[t-1]
    #   VMA[0] = price[0]
    # ------------------------------------------------------------------
    vma = pd.Series(index=close.index, dtype="float64")
    prev = np.nan

    # Because df_sorted is sorted, this loop walks oldest -> newest,
    # matching Thinkorswim's recursion order.
    for idx, price_val, c in zip(close.index, close.values, coeff.values):
        if np.isnan(price_val):
            vma[idx] = np.nan
            continue

        if np.isnan(prev):
            # First bar: initialize with price (historical data in TOS CompoundValue)
            prev = price_val
        else:
            c_val = 0.0 if np.isnan(c) else float(c)
            prev = c_val * price_val + (1.0 - c_val) * prev

        vma[idx] = prev

    # ------------------------------------------------------------------
    # Volume-weighted moving averages (VWMA) for fast / intermediate / slow
    # ------------------------------------------------------------------
    vwma_fast = (volume * close).rolling(fast_len).sum() / volume.rolling(fast_len).sum()
    vwma_int = (volume * close).rolling(int_len).sum() / volume.rolling(int_len).sum()
    vwma_slow = (volume * close).rolling(slow_len).sum() / volume.rolling(slow_len).sum()

    bullish = (vwma_fast > vwma_int) & (vwma_int > vwma_slow)
    bearish = (vwma_fast < vwma_int) & (vwma_int < vwma_slow)

    # ------------------------------------------------------------------
    # Stage classification (numeric)
    # ------------------------------------------------------------------
    stage = pd.Series(index=close.index, dtype="float64")

    accel = (close > vma) & bullish
    decel = (close < vma) & bearish

    stage[accel] = 2.0
    stage[decel] = -2.0

    rising = (vma > vma.shift(1)) & ~accel & ~decel
    falling = (vma < vma.shift(1)) & ~accel & ~decel

    stage[rising] = 1.0
    stage[falling] = -1.0

    # Everything else stays 0.0 (neutral / distribution)
    stage = stage.fillna(0.0)

    # Reindex back to the original df index just in case the caller passes
    # an unsorted frame (apply_core already sorts, so this is usually a no-op).
    return stage.reindex(df.index)
  
