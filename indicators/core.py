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


# Namespace+timeframe key
NT = Tuple[str, str]


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
"""
def indicator_trend_score(
    df: pd.DataFrame,
    fast_len: int,
    slow_len: int,
    slope_len: int,
    ma_type: str = "ema",
    **_,
) -> pd.Series:
    
    #Composite trend strength/direction score.
    
    if len(df) < max(fast_len, slow_len, slope_len):
        return pd.Series(index=df.index, dtype="float64")

    close = df["close"].astype(float)

    if ma_type == "sma":
        fast_ma = _sma(close, fast_len)
        slow_ma = _sma(close, slow_len)
    else:
        fast_ma = _ema(close, fast_len)
        slow_ma = _ema(close, slow_len)

    spread = (fast_ma - slow_ma) / close.replace(0, np.nan)
    slope = _rolling_slope(slow_ma, slope_len)

    spread_norm = np.tanh(spread.clip(-3, 3))
    slope_norm = np.tanh(slope.clip(-3, 3))

    score = 0.6 * spread_norm + 0.4 * slope_norm
    return score.astype(float)
"""

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
    "vol_regime": indicator_vol_regime,
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
