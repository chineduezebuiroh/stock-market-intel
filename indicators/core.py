from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Column sets / public constants
# ---------------------------------------------------------------------

PRICE_COLS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

percentile_floor           = float(33.0)
percentile_ceiling         = float(67.0)

# Namespace+timeframe key
NT = Tuple[str, str]

# Root/data paths for reading benchmark parquets (SPY/QQQ)
_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _ROOT / "data"

# Cache: (timeframe_name, symbol, length) -> full vol_MA Series (indexed by date)
_BENCHMARK_VOL_MA_CACHE: dict[tuple[str, str, int], pd.Series] = {}


@dataclass(frozen=True)
class IndicatorInstance:
    """
    A single configured indicator for a specific (namespace, timeframe).

    key:       instance key & column name (e.g. "ema_8", "atr_14", "trend_score")
    base_id:   generic indicator id (e.g. "ema", "atr", "trend_score")
    params:    dict of parameters passed to the indicator function
    """
    key: str
    base_id: str
    params: Mapping[str, object]


# ---------------------------------------------------------------------
# Low-level helpers (same as before, plus _sma)
# ---------------------------------------------------------------------
def _load_benchmark_vol_ma(
    timeframe_name: str,
    symbol: str,
    length: int,
) -> pd.Series:
    """
    Load volume SMA(length) for a benchmark symbol (e.g. SPY, QQQ)
    for a given timeframe, using the existing per-symbol parquet.

    Returns a Series indexed by date. Caller is responsible for
    reindexing it to match their own df.index if needed.
    """
    key = (timeframe_name, symbol, int(length))
    if key in _BENCHMARK_VOL_MA_CACHE:
        return _BENCHMARK_VOL_MA_CACHE[key]

    # For stocks namespace: data/timeframe=stocks_<tf>/ticker=<SYM>/data.parquet
    tf_dir = f"timeframe=stocks_{timeframe_name}"
    path = _DATA_DIR / tf_dir / f"ticker={symbol}" / "data.parquet"

    if not path.exists():
        # If benchmark parquet doesn't exist, return empty series
        s = pd.Series(dtype="float64")
        _BENCHMARK_VOL_MA_CACHE[key] = s
        return s

    df_bench = pd.read_parquet(path)
    if "volume" not in df_bench.columns or df_bench.empty:
        s = pd.Series(dtype="float64")
        _BENCHMARK_VOL_MA_CACHE[key] = s
        return s

    vol = df_bench["volume"].astype(float)
    vol_ma = vol.rolling(window=length, min_periods=length).mean()

    _BENCHMARK_VOL_MA_CACHE[key] = vol_ma
    return vol_ma


def _sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=length, min_periods=length).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Standard exponential moving average."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _wema(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder-style EMA: EMA with alpha = 1 / length.
    This is what many ATR/RSI style calcs use under the hood.
    """
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()


def _rolling_slope(series: pd.Series, length: int) -> pd.Series:
    """
    Rolling linear regression slope over 'length' bars.

    We fit y = a*x + b over the last N points and return 'a' (the slope).
    x is 0..N-1 so the scale is consistent across timeframes.
    """
    s = series.astype(float)

    if length <= 1:
        return pd.Series(index=s.index, dtype="float64")

    # Pre-compute x for the window length
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)

    def _slope_window(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        return cov_xy / x_var if x_var != 0 else np.nan

    return (
        s.rolling(window=length, min_periods=length)
        .apply(_slope_window, raw=True)
        .astype(float)
    )


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Average True Range (Wilder-style) over 'length' periods.
    Requires 'high', 'low', 'close' columns.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return _wema(tr, length)


# Define a function to calculate percentile rank of the last element
def _pctrank(x):
    # x is a Series representing the current window
    # x.iloc[-1] is the current value
    # len(x[x <= x.iloc[-1]]) counts elements less than or equal to the current value
    # len(x) is the total number of elements in the window
    return len(x[x <= x.iloc[-1]]) / len(x) * 100


def _linear_reg_curve(series: pd.Series, length: int) -> pd.Series:
    """
    Rolling linear regression curve over 'length' bars.

    For each window of size length, fit y = a*x + b (x = 0..length-1)
    and return the fitted value at the last x (x = length-1).

    This approximates Thinkorswim's LinearRegCurve behavior.
    """
    s = series.astype(float)

    if length <= 1:
        return pd.Series(index=s.index, dtype="float64")

    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)

    def _lr_last(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        if x_var == 0:
            return np.nan
        a = cov_xy / x_var
        b = y_mean - a * x_mean
        return a * x[-1] + b

    return (
        s.rolling(window=length, min_periods=length)
        .apply(_lr_last, raw=True)
        .astype(float)
    )

# ---------------------------------------------------------------------
# Primitive indicators (used in instance params)
# ---------------------------------------------------------------------

def indicator_sma(df: pd.DataFrame, length: int, src: str = "close", **_) -> pd.Series:
    return _sma(df[src].astype(float), length)


def indicator_ema(df: pd.DataFrame, length: int, src: str = "close", **_) -> pd.Series:
    return _ema(df[src].astype(float), length)


def indicator_wema(df: pd.DataFrame, length: int, src: str = "close", **_) -> pd.Series:
    return _wema(df[src].astype(float), length)


def indicator_vol_sma(df: pd.DataFrame, length: int, src: str = "volume", **_) -> pd.Series:
    return _sma(df[src].astype(float), length)


def indicator_atr(df: pd.DataFrame, length: int, **_) -> pd.Series:
    return _atr(df, length)


def indicator_slope(df: pd.DataFrame, length: int, src: str, **_) -> pd.Series:
    """
    Slope of an existing column (e.g. "ema_8") over a rolling window.
    """
    if src not in df.columns:
        # Not enough structure; return NaN, don't crash.
        return pd.Series(index=df.index, dtype="float64")
    return _rolling_slope(df[src].astype(float), length)


# ---------------------------------------------------------------------
# Composite indicators (same prototypes you just tested)
# ---------------------------------------------------------------------
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


def indicator_exh_abs_price_action(
    df: pd.DataFrame,
    percentile_floor: float = 33.0,
    percentile_ceiling: float = 67.0,
    lookback_period: int = 126,
    pinbar_scan_period: int = 2,
    pinbar_bar_check_count: int = 1,
    wick_adj_factor: float = 0.25,
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

    cond = (vol_pct_ceil >= pc) | (vol_pct_ceil.shift(1) >= pc)
    scan[cond] = 1.0

    # Reindex back to original df index (preserves caller's index order)
    return scan.reindex(df.index)


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
    scan[~rising_f & falling_f] = -1.0
    scan[squeeze_f] = 0.0

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


# ---------------------------------------------------------------------
# Indicator function registry
#   base_id -> function(df, **params) -> Series
# ---------------------------------------------------------------------

INDICATOR_FUNCS: Dict[str, Callable[..., pd.Series]] = {
    "sma": indicator_sma,
    "ema": indicator_ema,
    "wema": indicator_wema,
    "vol_sma": indicator_vol_sma,
    "atr": indicator_atr,
    "slope": indicator_slope,

    "wyckoff_stage": indicator_wyckoff_stage,
    "exh_abs_price_action": indicator_exh_abs_price_action,
    "significant_volume": indicator_significant_volume,
    "spy_qqq_volume_ma_ratio": indicator_spy_qqq_volume_ma_ratio,
    "movingavg_trend_cloud": indicator_movingavg_trend_cloud,
    "macdv": indicator_macdv,
    "ttm_squeeze_pro": indicator_ttm_squeeze_pro,
    "entry_signal": indicator_entry_signal,
}


# ---------------------------------------------------------------------
# Config-backed storage (built by initialize_indicator_engine)
# ---------------------------------------------------------------------

_STORAGE_PROFILES: Dict[NT, List[IndicatorInstance]] = {}
_PROFILE_DEFS: Dict[str, List[str]] = {}  # profile_name -> list of instance keys
_PARAMS: Dict[NT, Dict[str, Dict[str, object]]] = {}  # (ns, tf) -> instance_key -> param dict
_INITIALIZED: bool = False
_CONFIG_DIR: Path | None = None


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _load_indicator_profiles(config_dir: Path) -> Dict[str, List[str]]:
    data = _load_yaml(config_dir / "indicator_profiles.yaml")
    profiles = data.get("profiles", {})
    return {name: list(keys) for name, keys in profiles.items()}


def _load_indicator_params(config_dir: Path) -> Dict[NT, Dict[str, Dict[str, object]]]:
    data = _load_yaml(config_dir / "indicator_params.yaml")
    indicators = data.get("indicators", {})

    result: Dict[NT, Dict[str, Dict[str, object]]] = {}
    for namespace, by_tf in indicators.items():
        for timeframe, inst_map in by_tf.items():
            key = (namespace, timeframe)
            result[key] = {}
            for instance_key, cfg in inst_map.items():
                base_id = cfg.get("id")
                if base_id is None:
                    raise ValueError(
                        f"indicator_params: {namespace}/{timeframe}/{instance_key} missing 'id'"
                    )
                result[key][instance_key] = dict(cfg)
    return result


def _load_combos(config_dir: Path) -> Dict[str, dict]:
    return _load_yaml(config_dir / "multi_timeframe_combos.yaml")


def _build_storage_profiles(
    profiles: Dict[str, List[str]],
    params: Dict[NT, Dict[str, Dict[str, object]]],
    combos_data: Dict[str, dict],
) -> Dict[NT, List[IndicatorInstance]]:
    storage: Dict[NT, List[IndicatorInstance]] = {}

    ROLE_FIELDS = [
        ("lower_tf", "lower_profile"),
        ("middle_tf", "middle_profile"),
        ("upper_tf", "upper_profile"),
    ]

    for namespace, combos in combos_data.items():
        for combo_name, cfg in combos.items():
            if not isinstance(cfg, dict):
                continue

            for tf_field, profile_field in ROLE_FIELDS:
                timeframe = cfg.get(tf_field)
                profile_name = cfg.get(profile_field)

                if not timeframe or not profile_name:
                    continue

                nt = (namespace, timeframe)
                profile_keys = profiles.get(profile_name)
                if profile_keys is None:
                    raise KeyError(
                        f"Combo '{namespace}/{combo_name}' references unknown profile '{profile_name}'"
                    )

                tf_params = params.get(nt, {})
                if not tf_params:
                    raise KeyError(
                        f"No indicator_params defined for namespace={namespace}, timeframe={timeframe}"
                    )

                existing_keys = {inst.key for inst in storage.get(nt, [])}
                instances_for_nt: List[IndicatorInstance] = storage.get(nt, []).copy()

                for instance_key in profile_keys:
                    if instance_key in existing_keys:
                        continue

                    cfg_for_key = tf_params.get(instance_key)
                    if cfg_for_key is None:
                        raise KeyError(
                            f"indicator_params missing for {namespace}/{timeframe}/{instance_key}"
                        )

                    base_id = cfg_for_key.get("id")
                    if base_id not in INDICATOR_FUNCS:
                        raise KeyError(
                            f"Unknown base indicator id '{base_id}' "
                            f"for {namespace}/{timeframe}/{instance_key}"
                        )

                    params_without_id = {
                        k: v for k, v in cfg_for_key.items() if k != "id"
                    }

                    instances_for_nt.append(
                        IndicatorInstance(
                            key=instance_key,
                            base_id=base_id,
                            params=params_without_id,
                        )
                    )
                    existing_keys.add(instance_key)

                storage[nt] = instances_for_nt

    return storage


def initialize_indicator_engine(config_dir: str | Path = "config") -> None:
    """
    Load indicator_profiles.yaml, indicator_params.yaml, multi_timeframe_combos.yaml
    and build storage profiles for each (namespace, timeframe).
    """
    global _INITIALIZED, _CONFIG_DIR, _STORAGE_PROFILES, _PROFILE_DEFS, _PARAMS

    _CONFIG_DIR = Path(config_dir)
    _PROFILE_DEFS = _load_indicator_profiles(_CONFIG_DIR)
    _PARAMS = _load_indicator_params(_CONFIG_DIR)
    combos = _load_combos(_CONFIG_DIR)

    _STORAGE_PROFILES = _build_storage_profiles(
        profiles=_PROFILE_DEFS,
        params=_PARAMS,
        combos_data=combos,
    )

    _INITIALIZED = True


def _ensure_initialized() -> None:
    if not _INITIALIZED:
        initialize_indicator_engine(_CONFIG_DIR or "config")


# ---------------------------------------------------------------------
# Core application API
# ---------------------------------------------------------------------

def apply_core(df: pd.DataFrame, namespace: str, timeframe: str) -> pd.DataFrame:
    """
    Attach all indicators needed for this (namespace, timeframe),
    based on profiles+params+combos config.

    ❗ Invariants:
      - No MultiIndex columns are created.
      - Index is preserved; assumed naive and ascending.
      - Columns remain flat strings.
      - Failures degrade to NaN instead of raising.
    """
    _ensure_initialized()

    nt = (namespace, timeframe)
    instances = _STORAGE_PROFILES.get(nt, [])

    out = df.copy().sort_index()

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        print(f"[WARN] apply_core: missing columns {missing}, skipping indicators.")
        return out

    for inst in instances:
        func = INDICATOR_FUNCS[inst.base_id]
        try:
            series = func(out, **inst.params)
        except Exception as e:
            print(
                f"[WARN] indicator {inst.base_id} ({inst.key}) failed for "
                f"{namespace}/{timeframe}: {e}"
            )
            series = pd.Series(index=out.index, dtype="float64")
        out[inst.key] = series

    out.columns = out.columns.astype(str)
    return out


def get_snapshot_base_cols(namespace: str, timeframe: str) -> List[str]:
    """
    For a given (namespace, timeframe), return:
        [open, high, low, close, adj_close, volume, ...all indicator cols...]
    """
    _ensure_initialized()
    nt = (namespace, timeframe)
    instances = _STORAGE_PROFILES.get(nt, [])
    indicator_cols = [inst.key for inst in instances]
    return PRICE_COLS + indicator_cols
