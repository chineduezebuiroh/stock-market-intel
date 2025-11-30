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


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Standard exponential moving average."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _wema(series: pd.Series, length: int) -> pd.Series:
    """Wilder-style EMA, alpha = 1 / length."""
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()


def _rolling_slope(series: pd.Series, length: int) -> pd.Series:
    """
    Rolling linear regression slope over 'length' bars.
    (Move your existing implementation here.)
    """
    # paste your current _rolling_slope body
    ...


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Average True Range (Wilder-style) over 'length' periods.
    (Move your existing implementation here.)
    """
    # paste your current _atr body
    ...


def _pctrank(window: pd.Series) -> float:
    """
    Percentile rank (0-100) of the last element within the window.
    (Use the version we wrote for significant_volume.)
    """
    w = window.dropna()
    if len(w) == 0:
        return np.nan
    return w.rank(pct=True).iloc[-1] * 100.0


def _linear_reg_curve(series: pd.Series, length: int) -> pd.Series:
    """
    Rolling linear regression curve over 'length' bars.
    (Use the version we wrote for TTM Squeeze Pro.)
    """
    # paste your current _linear_reg_curve body
    ...


def _load_benchmark_vol_ma(
    timeframe_name: str,
    symbol: str,
    length: int,
) -> pd.Series:
    """
    Load SMA(volume, length) for a benchmark symbol (SPY, QQQ) for a timeframe.
    (Use the version you already have.)
    """
    # paste your current _load_benchmark_vol_ma body
    ...
