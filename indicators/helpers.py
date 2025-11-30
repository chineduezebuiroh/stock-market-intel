# indicators/helpers.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# Root / data paths for benchmarks (SPY/QQQ) etc.
_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _ROOT / "data"

# Cache: (timeframe_name, symbol, length) -> Series
_BENCHMARK_VOL_MA_CACHE: dict[tuple[str, str, int], pd.Series] = {}

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
