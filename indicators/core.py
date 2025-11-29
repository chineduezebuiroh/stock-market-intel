from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Column sets / public constants
# -----------------------------------------------------------------------------
# Base OHLCV columns expected on every per-symbol parquet
PRICE_COLS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

# Core indicator columns produced by apply_core
CORE_INDICATOR_COLS: list[str] = [
    "sma_8",
    "ema_8",
    "ema_21",
    "wema_14",
    "ema_8_slope",
    "ema_21_slope",
    "volume_sma_20",
    "atr_14",
]

# For snapshots: base price + indicator columns
SNAPSHOT_BASE_COLS: list[str] = PRICE_COLS + CORE_INDICATOR_COLS


def get_snapshot_base_cols() -> list[str]:
    """
    Public helper so snapshot builders don't hardcode base column names.

    Returns all columns that must be present in a snapshot row
    (excluding the 'symbol' column, which is added by the caller).
    """
    return list(SNAPSHOT_BASE_COLS)


# -----------------------------------------------------------------------------
# Length presets per timeframe (can be expanded later)
# -----------------------------------------------------------------------------
# You can customize these per timeframe later. For now:
# - "default" matches your existing settings exactly.
CORE_LENGTH_PRESETS: dict[str, dict[str, int]] = {
    "default": {
        "sma_len": 8,
        "ema_short_len": 8,
        "ema_long_len": 21,
        "wema_len": 14,
        "vol_sma_len": 20,
        "atr_len": 14,
        "slope_len": 3,   # rolling window for slope
    },
    # Example future overrides:
    # "weekly": {
    #     "sma_len": 8,
    #     "ema_short_len": 8,
    #     "ema_long_len": 21,
    #     "wema_len": 14,
    #     "vol_sma_len": 20,
    #     "atr_len": 14,
    #     "slope_len": 3,
    # },
    # "intraday_130m": {
    #     "sma_len": 8,
    #     "ema_short_len": 8,
    #     "ema_long_len": 21,
    #     "wema_len": 14,
    #     "vol_sma_len": 20,
    #     "atr_len": 14,
    #     "slope_len": 3,
    # },
}


def _resolve_lengths(params: dict | None) -> dict[str, int]:
    """
    Resolve indicator lengths based on params.

    Right now this just switches on params.get("timeframe").
    You can later key this on namespace, instrument type, etc.
    """
    if params is None:
        params = {}

    tf = params.get("timeframe", "default")
    cfg = CORE_LENGTH_PRESETS.get(tf)

    if cfg is None:
        # Fallback to default if unknown timeframe
        cfg = CORE_LENGTH_PRESETS["default"]

    return cfg


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Composite indicators (store-only outputs; internal EMAs/ATR/etc.)
# ---------------------------------------------------------------------

def indicator_trend_score(
    df: pd.DataFrame,
    fast_len: int,
    slow_len: int,
    slope_len: int,
    ma_type: str = "ema",
    **_,
) -> pd.Series:
    """
    Composite trend strength/direction score.

    Rough prototype:
      - compute fast & slow MA of close
      - compute their difference normalized by price
      - compute slope of slow MA
      - combine into a score ~[-1, 1]

    You can replace all of this with your actual logic later.
    """
    if len(df) < max(fast_len, slow_len, slope_len):
        # Not enough history; return all-NaN
        return pd.Series(index=df.index, dtype="float64")

    close = df["close"].astype(float)

    if ma_type == "sma":
        fast_ma = _sma(close, fast_len)
        slow_ma = _sma(close, slow_len)
    else:
        fast_ma = _ema(close, fast_len)
        slow_ma = _ema(close, slow_len)

    # Price-relative MA spread
    spread = (fast_ma - slow_ma) / close.replace(0, np.nan)

    # Slope of slow MA
    slope = _rolling_slope(slow_ma, slope_len)

    # Simple normalization & combination
    # (clip to [-3,3], then squish with tanh to [-1,1])
    spread_norm = np.tanh(spread.clip(-3, 3))
    slope_norm = np.tanh(slope.clip(-3, 3))

    score = 0.6 * spread_norm + 0.4 * slope_norm
    return score.astype(float)


def indicator_vol_regime(
    df: pd.DataFrame,
    atr_len: int,
    lookback: int,
    low_pct: float,
    high_pct: float,
    **_,
) -> pd.Series:
    """
    Volatility regime classification using ATR as % of price.

    Rough prototype:
      - compute ATR(atr_len)
      - atr_pct = ATR / close
      - smooth atr_pct with SMA(lookback)
      - classify:
          atr_smoothed < low_pct  -> -1 (low vol)
          low_pct <= ... <= high_pct -> 0 (normal)
          atr_smoothed > high_pct -> +1 (high vol)
    """
    min_len = max(atr_len, lookback)
    if len(df) < min_len:
        return pd.Series(index=df.index, dtype="float64")

    close = df["close"].astype(float)
    atr = _atr(df, atr_len)
    atr_pct = atr / close.replace(0, np.nan)

    atr_smoothed = _sma(atr_pct, lookback)

    regime = pd.Series(index=df.index, dtype="float64")

    regime[atr_smoothed < low_pct] = -1.0
    regime[(atr_smoothed >= low_pct) & (atr_smoothed <= high_pct)] = 0.0
    regime[atr_smoothed > high_pct] = 1.0

    return regime


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

    Rough prototype:
      - signal = 1 if:
          trend_score >= trend_min
          vol_regime <= max_vol_regime (e.g. <= 0, so low/normal vol)
        else 0

    Requires that df already has 'trend_score' and 'vol_regime' columns
    for this (namespace, timeframe).
    """
    if "trend_score" not in df.columns or "vol_regime" not in df.columns:
        # If deps aren't ready, just return NaN; the engine won't crash.
        return pd.Series(index=df.index, dtype="float64")

    trend = df["trend_score"].astype(float)
    vol = df["vol_regime"].astype(float)

    cond = (trend >= trend_min) & (vol <= max_vol_regime)
    signal = cond.astype(float)  # 1.0 or 0.0

    return signal



# -----------------------------------------------------------------------------
# Core indicator application
# -----------------------------------------------------------------------------
def apply_core(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Attach core indicators to a per-symbol OHLCV dataframe.

    Expected input columns (per-symbol parquet):
        open, high, low, close, adj_close, volume

    Output (additional) columns:
        sma_8
        ema_8
        ema_21
        wema_14
        ema_8_slope
        ema_21_slope
        volume_sma_20
        atr_14

    'params' is reserved for per-timeframe overrides, e.g.:
        params = {"timeframe": "daily"}

    ❗ Invariants:
        - No MultiIndex columns are ever created.
        - Index is preserved and assumed already normalized (naive, time-ascending).
        - Columns remain flat strings.
    """
    # Work on a copy to avoid side effects
    out = df.copy()

    # Ensure sorted index (time ascending) for rolling operations
    out = out.sort_index()

    # Basic sanity: require key columns
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        # If the structure is wrong, just return as-is (no indicators)
        # You can make this stricter later if you want.
        print(f"[WARN] apply_core: missing columns {missing}, skipping indicator calc.")
        return out

    # Resolve lengths (this preserves your old defaults when params is None)
    lengths = _resolve_lengths(params)
    sma_len = lengths["sma_len"]
    ema_short_len = lengths["ema_short_len"]
    ema_long_len = lengths["ema_long_len"]
    wema_len = lengths["wema_len"]
    vol_sma_len = lengths["vol_sma_len"]
    atr_len = lengths["atr_len"]
    slope_len = lengths["slope_len"]

    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    # -------------------------------------------------------------------------
    # Price-based moving averages
    # -------------------------------------------------------------------------
    out["sma_8"] = close.rolling(window=sma_len, min_periods=sma_len).mean()
    out["ema_8"] = _ema(close, ema_short_len)
    out["ema_21"] = _ema(close, ema_long_len)
    out["wema_14"] = _wema(close, wema_len)

    # Slopes (trend strength / direction)
    out["ema_8_slope"] = _rolling_slope(out["ema_8"], slope_len)
    out["ema_21_slope"] = _rolling_slope(out["ema_21"], slope_len)

    # -------------------------------------------------------------------------
    # Volume-based
    # -------------------------------------------------------------------------
    out["volume_sma_20"] = volume.rolling(window=vol_sma_len, min_periods=vol_sma_len).mean()

    # -------------------------------------------------------------------------
    # Volatility-based
    # -------------------------------------------------------------------------
    out["atr_14"] = _atr(out, atr_len)

    # Ensure flat string columns
    out.columns = out.columns.astype(str)

    return out
