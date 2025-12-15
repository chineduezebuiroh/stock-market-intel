from __future__ import annotations

# indicators/composite_using_percentiles.py
import numpy as np
import pandas as pd

from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve

# ------------------------------------------------------------------
# Set Global Constants
# ------------------------------------------------------------------
global_percentile_floor = float(33.0)
global_percentile_ceiling = float(67.0)


def indicator_exh_abs_price_action(
    df: pd.DataFrame,
    percentile_floor: float = 33.0,
    percentile_ceiling: float = 67.0,
    lookback_period: int = 126,
    pinbar_scan_period: int = 2,
    pinbar_bar_check_count: int = 1,
    wick_adj_factor: float = 0.33,
    strict_candlebody_check: bool = False,
    z: int = 0,
    **_,
) -> pd.Series:

    """
    Exh_Abs_Price_Action (Thinkorswim port).

    Numeric encoding (matches the TOS 'scan' output):
        z = 0 (signal on current bar):
          +1  = ExhPinbarBot or AbsPinbarBot
          +2  = EngulfingCandleBot
          -1  = ExhPinbarTop or AbsPinbarTop
          -2  = EngulfingCandleTop
           0  = no pattern

        z = 1 (signal on bar after pattern, with confirmation on current bar):
          +1  = prior Exh/AbsPinbarBot and low[0] >= low[1]
          +2  = prior EngulfingCandleBot and open >= close[1] and high > high[1]
          -1  = prior Exh/AbsPinbarTop and high[0] <= high[1]
          -2  = prior EngulfingCandleTop and open <= close[1] and low < low[1]
           0  = otherwise

    Returns a float Series with values in {-2, -1, 0, 1, 2}.
    """

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.Series(index=df.index, dtype="float64")

    # Ensure chronological order (oldest -> newest)
    df_sorted = df.sort_index()

    open_ = df_sorted["open"].astype(float)
    high = df_sorted["high"].astype(float)
    low = df_sorted["low"].astype(float)
    close = df_sorted["close"].astype(float)
    volume = df_sorted["volume"].astype(float)

    # ------------------------------------------------------------------
    # Pinbar wick logic
    # ------------------------------------------------------------------
    max_oc = open_.combine(close, np.maximum)
    min_oc = open_.combine(close, np.minimum)

    bearish_pinbar_wick = (high - max_oc) > wick_adj_factor * (min_oc - low)
    bullish_pinbar_wick = (min_oc - low) > wick_adj_factor * (high - max_oc)

    # ------------------------------------------------------------------
    # Candlebody "adjusted" magnitude
    # candlebody_adj = (|close-open| * 100) ^ candlebody_ratio
    # where candlebody_ratio = |close-open| / (high-low), but 0 if high == low
    # ------------------------------------------------------------------
    candlebody = (close - open_).abs()
    hl_range = (high - low)

    with np.errstate(divide="ignore", invalid="ignore"):
        candlebody_ratio = np.where(hl_range == 0, 0.0, candlebody / hl_range)

    candlebody_adj = np.power(candlebody * 100.0, candlebody_ratio)
    candlebody_adj = np.where(hl_range == 0, 0.0, candlebody_adj)
    candlebody_adj = pd.Series(candlebody_adj, index=df_sorted.index)

    # ------------------------------------------------------------------
    # Percentiles over lookback_period for candlebody_adj and volume
    # TOS fold: count of prior bars where current > prior, normalized by L.
    # ------------------------------------------------------------------
    L = int(lookback_period)

    cb_count = pd.Series(0.0, index=df_sorted.index)
    vol_count = pd.Series(0.0, index=df_sorted.index)

    # Apply the rolling percentile calculation
    window_size = L  # Example window size
    
    candlebody_percentile = candlebody_adj.rolling(window=window_size).apply(_pctrank, raw=False)
    volume_percentile = volume.rolling(window=window_size).apply(_pctrank, raw=False)
    
    # RoundDown / RoundUp to integer percent
    cb_pct_floor = np.floor(candlebody_percentile)
    vol_pct_floor = np.floor(volume_percentile)
    vol_pct_ceil = np.ceil(volume_percentile)

    # ------------------------------------------------------------------
    # Candlebody strict/lax checks
    # ------------------------------------------------------------------
    if strict_candlebody_check:
        bearish_pinbar_check = min_oc  # stricter: pinbar wick off the body low/high
        bullish_pinbar_check = max_oc
    else:
        bearish_pinbar_check = max_oc  # laxer: allow body to penetrate prior bodies
        bullish_pinbar_check = min_oc

    prev1_min = open_.shift(1).combine(close.shift(1), np.minimum)
    prev2_min = open_.shift(2).combine(close.shift(2), np.minimum)
    prev3_min = open_.shift(3).combine(close.shift(3), np.minimum)

    prev1_max = open_.shift(1).combine(close.shift(1), np.maximum)
    prev2_max = open_.shift(2).combine(close.shift(2), np.maximum)
    prev3_max = open_.shift(3).combine(close.shift(3), np.maximum)

    count_bearish_check = (
        (bearish_pinbar_check >= prev1_max).astype(int)
        + (bearish_pinbar_check >= prev2_max).astype(int)
        + (bearish_pinbar_check >= prev3_max).astype(int)
    )

    count_bullish_check = (
        (bullish_pinbar_check <= prev1_min).astype(int)
        + (bullish_pinbar_check <= prev2_min).astype(int)
        + (bullish_pinbar_check <= prev3_min).astype(int)
    )

    # ------------------------------------------------------------------
    # Exhaustion / Absolute Pinbar patterns
    # ------------------------------------------------------------------
    pf = float(percentile_floor)
    pc = float(percentile_ceiling)

    cond_exh_common = (
        (pf - vol_pct_floor) + (pf - cb_pct_floor) >= 0
    )

    highest_high_prev = high.rolling(pinbar_scan_period).max().shift(1)
    lowest_low_prev = low.rolling(pinbar_scan_period).min().shift(1)

    exh_pinbar_top = (
        cond_exh_common
        & (high >= highest_high_prev)
        & (count_bearish_check >= pinbar_bar_check_count)
        & bearish_pinbar_wick
    )

    exh_pinbar_bot = (
        cond_exh_common
        & (low <= lowest_low_prev)
        & (count_bullish_check >= pinbar_bar_check_count)
        & bullish_pinbar_wick
    )

    # Absolute Pinbars (volume strongly high)
    cond_abs_common = (
        (vol_pct_ceil - pc) + (pf - cb_pct_floor) >= 0
    )

    abs_pinbar_top = (
        cond_abs_common
        & (high >= highest_high_prev)
        & (count_bearish_check >= pinbar_bar_check_count)
        & bearish_pinbar_wick
    )

    abs_pinbar_bot = (
        cond_abs_common
        & (low <= lowest_low_prev)
        & (count_bullish_check >= pinbar_bar_check_count)
        & bullish_pinbar_wick
    )

    # ------------------------------------------------------------------
    # Engulfing candle patterns
    # ------------------------------------------------------------------
    hh_prev1 = highest_high_prev
    hh_prev2 = high.rolling(pinbar_scan_period).max().shift(2)

    ll_prev1 = lowest_low_prev
    ll_prev2 = low.rolling(pinbar_scan_period).min().shift(2)

    # Engulfing top
    engulf_top_cond1 = (high >= hh_prev1) | (high.shift(1) >= hh_prev2)
    engulf_top_cond2 = (count_bearish_check >= pinbar_bar_check_count)
    engulf_top_cond3 = (
        (open_ >= close.shift(1)) & (close < open_.shift(1))
        & (open_ > close) & (open_.shift(1) <= close.shift(1))
        & ((open_ >= high.shift(1)) | (close <= low.shift(1)))
    )
    engulfing_top = engulf_top_cond1 & engulf_top_cond2 & engulf_top_cond3

    # Engulfing bottom
    engulf_bot_cond1 = (low <= ll_prev1) | (low.shift(1) <= ll_prev2)
    engulf_bot_cond2 = (count_bullish_check >= pinbar_bar_check_count)
    engulf_bot_cond3 = (
        (open_ <= close.shift(1)) & (close > open_.shift(1))
        & (open_ < close) & (open_.shift(1) >= close.shift(1))
        & ((open_ <= low.shift(1)) | (close >= high.shift(1)))
    )
    engulfing_bot = engulf_bot_cond1 & engulf_bot_cond2 & engulf_bot_cond3

    # ------------------------------------------------------------------
    # Final scan values (z = 0: current bar; z = 1: prior bar + confirmation)
    # ------------------------------------------------------------------
    scan = pd.Series(0.0, index=df_sorted.index)

    if int(z) == 0:
        scan[exh_pinbar_bot | abs_pinbar_bot] = 1.0
        scan[engulfing_bot] = 2.0
        scan[exh_pinbar_top | abs_pinbar_top] = -1.0
        scan[engulfing_top] = -2.0

    elif int(z) == 1:
        prior_pinbar_bot = exh_pinbar_bot.shift(1) | abs_pinbar_bot.shift(1)
        prior_pinbar_top = exh_pinbar_top.shift(1) | abs_pinbar_top.shift(1)
        prior_engulf_bot = engulfing_bot.shift(1)
        prior_engulf_top = engulfing_top.shift(1)

        cond1 = prior_pinbar_bot & (low >= low.shift(1))
        cond2 = prior_engulf_bot & (open_ >= close.shift(1)) & (high > high.shift(1))
        cond3 = prior_pinbar_top & (high <= high.shift(1))
        cond4 = prior_engulf_top & (open_ <= close.shift(1)) & (low < low.shift(1))

        scan[cond1] = 1.0
        scan[cond2] = 2.0
        scan[cond3] = -1.0
        scan[cond4] = -2.0

    # Reindex back to original df index (preserves caller's index order)
    return scan.reindex(df.index)


def indicator_significant_volume(
    df: pd.DataFrame,
    percentile_ceiling: float = 67.0,
    lookback_period: int = 126,
    z: int = 0,
    **_,
) -> pd.Series:

    """
    Significant_Volume (Thinkorswim port).

    Numeric encoding (matches the TOS 'scan' output):
        1 = current or prior volume percentile >= percentile_ceiling
        0 = both current and prior volume percentile < percentile_ceiling

    Returns a float Series with values in {0, 1}.
    """

    required = {"volume"}
    if not required.issubset(df.columns):
        return pd.Series(index=df.index, dtype="float64")

    # Ensure chronological order (oldest -> newest)
    df_sorted = df.sort_index()
    volume = df_sorted["volume"].astype(float)

    # ------------------------------------------------------------------
    # Percentiles over lookback_period for volume
    # ------------------------------------------------------------------
    L = int(lookback_period)
    
    # Apply the rolling percentile calculation
    window_size = L  # Example window size
    
    volume_percentile = volume.rolling(window=window_size).apply(_pctrank, raw=False)
    # RoundUp to integer percent
    vol_pct_ceil = np.ceil(volume_percentile)

    # ------------------------------------------------------------------
    # Final scan values
    # ------------------------------------------------------------------
    pc = float(percentile_ceiling)
    scan = pd.Series(0.0, index=df_sorted.index)

    z_int = int(z)
    vol_pct_ceil_z = vol_pct_ceil.shift(z_int)
    vol_pct_ceil_z1 = vol_pct_ceil.shift(z_int+1)

    cond = (vol_pct_ceil_z >= pc) | (vol_pct_ceil_z1 >= pc)
    scan[cond] = 1.0

    # Reindex back to original df index (preserves caller's index order)
    return scan.reindex(df.index)
