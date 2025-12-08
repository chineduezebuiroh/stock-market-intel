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
#from .helpers import _load_benchmark_vol_ma, _sma, _ema, _wema, _rolling_slope, _atr, _pctrank, _linear_reg_curve

# ---------------------------------------------------------------------
# Primitive indicators (used in instance params)
# ---------------------------------------------------------------------
"""
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
    
    #Slope of an existing column (e.g. "ema_8") over a rolling window.
    
    if src not in df.columns:
        # Not enough structure; return NaN, don't crash.
        return pd.Series(index=df.index, dtype="float64")
    return _rolling_slope(df[src].astype(float), length)
"""

# ---------------------------------------------------------------------
# Composite indicators (same prototypes you just tested)
# ---------------------------------------------------------------------
from .composite_wyckoff_stage import indicator_wyckoff_stage
from .composite_using_percentiles import (
    indicator_exh_abs_price_action,
    indicator_significant_volume,
)
from .composite_spy_qqq_volume_ma_ratio import indicator_spy_qqq_volume_ma_ratio
from .composite_movingavg_trend_cloud import indicator_movingavg_trend_bullish, indicator_movingavg_trend_bearish
from .composite_macdv import indicator_macdv, indicator_macdv_guardrail
from .composite_ttm_squeeze_pro import indicator_ttm_squeeze_pro, indicator_entry_signal

# ---------------------------------------------------------------------
# Indicator function registry
#   base_id -> function(df, **params) -> Series
# ---------------------------------------------------------------------

INDICATOR_FUNCS: Dict[str, Callable[..., pd.Series]] = {
    #"sma": indicator_sma,
    #"ema": indicator_ema,
    #"wema": indicator_wema,
    #"vol_sma": indicator_vol_sma,
    #"atr": indicator_atr,
    #"slope": indicator_slope,

    "wyckoff_stage": indicator_wyckoff_stage,
    "exh_abs_price_action": indicator_exh_abs_price_action,
    "significant_volume": indicator_significant_volume,
    "spy_qqq_volume_ma_ratio": indicator_spy_qqq_volume_ma_ratio,
    "movingavg_trend_bullish": indicator_movingavg_trend_bullish,
    "movingavg_trend_bearish": indicator_movingavg_trend_bearish,
    "macdv": indicator_macdv,
    "macdv_guard": indicator_macdv_guardrail,
    "ttm_squeeze_pro": indicator_ttm_squeeze_pro,
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
