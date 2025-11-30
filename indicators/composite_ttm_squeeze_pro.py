from __future__ import annotations

# indicators/composite_ttm_squeeze_pro.py
import numpy as np
import pandas as pd

from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve


def indicator_ttm_squeeze_pro(
    df: pd.DataFrame,
    study_length: int = 20,
    strict_check: bool = True,
    z: int = 0,
    **_,
) -> pd.Series:
    """
    TTM_Squeeze_Pro (Thinkorswim-style port).

    Logic:
      - Compute Bollinger Bands (length = study_length, default 2 std dev).
      - Compute Keltner Channels for 3 factors: 1.0 (orange), 1.5 (red), 2.0 (black).
      - Determine "dot" checks based on whether BB is inside KC (strict AND vs lax OR).
      - Compute Delta_Plot via LinearRegCurve of:
            Delta = close - (DonchianMid + SMA(close, length)) / 2
      - Scan output (with bar offset z):
            1  = Delta_Plot rising (short-term up bias)
           -1  = Delta_Plot falling (short-term down bias)
            0  = any squeeze dot active (orange/red/black) at z
           NaN = otherwise

    Returns a float Series with values in {1, -1, 0, NaN}.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        return pd.Series(index=df.index, dtype="float64")

    df_sorted = df.sort_index()
    close = df_sorted["close"].astype(float)
    high = df_sorted["high"].astype(float)
    low = df_sorted["low"].astype(float)

    L = int(study_length)
    if len(df_sorted) < L:
        return pd.Series(index=df.index, dtype="float64")

    # ------------------------------------------------------------------
    # Bollinger Bands (basis = SMA(close, L), std dev = 2 * stdev)
    # ------------------------------------------------------------------
    basis_bb = close.rolling(window=L, min_periods=L).mean()
    stdev_bb = close.rolling(window=L, min_periods=L).std(ddof=0)
    ub = basis_bb + 2.0 * stdev_bb
    lb = basis_bb - 2.0 * stdev_bb

    # ------------------------------------------------------------------
    # Keltner Channels: basis = EMA(close, L), bands = basis ± factor * ATR(L)
    # ------------------------------------------------------------------
    atr_L = _atr(df_sorted, L)
    basis_kc = _ema(close, L)

    def _kc_band(factor: float):
        upper = basis_kc + factor * atr_L
        lower = basis_kc - factor * atr_L
        return upper, lower

    ub_kc_orange, lb_kc_orange = _kc_band(1.0)
    ub_kc_red, lb_kc_red = _kc_band(1.5)
    ub_kc_black, lb_kc_black = _kc_band(2.0)

    # ------------------------------------------------------------------
    # Dot checks (strict AND vs lax OR)
    # ------------------------------------------------------------------
    if strict_check:
        orange_dot = (ub < ub_kc_orange) & (lb > lb_kc_orange)
        red_dot = (ub < ub_kc_red) & (lb > lb_kc_red)
        black_dot = (ub < ub_kc_black) & (lb > lb_kc_black)
    else:
        orange_dot = (ub < ub_kc_orange) | (lb > lb_kc_orange)
        red_dot = (ub < ub_kc_red) | (lb > lb_kc_red)
        black_dot = (ub < ub_kc_black) | (lb > lb_kc_black)

    # ------------------------------------------------------------------
    # Delta_Plot via LinearRegCurve of Delta
    # ------------------------------------------------------------------
    donchian_mid = (high.rolling(L).max() + low.rolling(L).min()) / 2.0
    sma_close = close.rolling(window=L, min_periods=L).mean()

    avg_line = (donchian_mid + sma_close) / 2.0
    delta = close - avg_line

    delta_plot = _linear_reg_curve(delta, L)

    # ------------------------------------------------------------------
    # Offsets: Delta_Plot[z], [z+1], [z+2]
    # ------------------------------------------------------------------
    z_int = int(z)
    dp_z = delta_plot.shift(z_int)
    dp_z1 = delta_plot.shift(z_int + 1)
    dp_z2 = delta_plot.shift(z_int + 2)

    # Rising / falling checks (loosely "momentum" or slope direction)
    rising = (dp_z > dp_z1) | (dp_z1 > dp_z2)
    falling = (dp_z < dp_z1) | (dp_z1 < dp_z2)

    # Squeeze dots at bar z
    orange_z = orange_dot.shift(z_int)
    red_z = red_dot.shift(z_int)
    black_z = black_dot.shift(z_int)
    squeeze_z = (orange_z | red_z | black_z)

    # ------------------------------------------------------------------
    # Final scan output, preserving TOS priority:
    #   if rising then 1
    #   else if falling then -1
    #   else if any dot then 0
    #   else NaN
    # ------------------------------------------------------------------
    scan = pd.Series(np.nan, index=df_sorted.index, dtype="float64")

    rising_f = rising.fillna(False)
    falling_f = falling.fillna(False)
    squeeze_f = squeeze_z.fillna(False)

    scan[rising_f] = 1.0
    scan[falling_f] = -1.0
    scan[(rising_f & falling_f) | squeeze_f] = 0.0

    return scan.reindex(df.index)



def indicator_entry_signal(
    df: pd.DataFrame,
    trend_min: float,
    max_vol_regime: float = 0.0,
    **_,
) -> pd.Series:
    """
    Entry signal (long bias) based on:
      - trend_score column already computed
      - vol_regime column already computed
    """
    if "trend_score" not in df.columns or "vol_regime" not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    trend = df["trend_score"].astype(float)
    vol = df["vol_regime"].astype(float)

    cond = (trend >= trend_min) & (vol <= max_vol_regime)
    signal = cond.astype(float)

    return signal
