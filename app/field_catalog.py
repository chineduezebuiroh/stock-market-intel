from __future__ import annotations

# app/field_catalog.py

from dataclasses import dataclass
from typing import List, Tuple


# -----------------------------
# Debug panel field definition
# -----------------------------
@dataclass(frozen=True)
class FieldSpec:
    suffix: str
    label: str


DEBUG_FIELDS: List[FieldSpec] = [
    FieldSpec("close", "Close"),
    FieldSpec("volume", "Volume"),
    FieldSpec("wyckoff_stage", "Wyckoff Stage"),
    FieldSpec("exh_abs_pa_current_bar", "Exh/Abs (current)"),
    FieldSpec("exh_abs_pa_prior_bar", "Exh/Abs (prior)"),
    FieldSpec("sig_vol_current_bar", "Sig Vol (current)"),
    FieldSpec("sig_vol_prior_bar", "Sig Vol (prior)"),
    FieldSpec("spy_qqq_vol_ma_ratio", "SPY/QQQ Vol Ratio"),
    FieldSpec("ma_trend_bullish", "MA Trend Bull"),
    FieldSpec("ma_trend_bearish", "MA Trend Bear"),
    FieldSpec("macdv_core_bull", "MACDV Core Bull"),
    FieldSpec("macdv_core_bear", "MACDV Core Bear"),
    FieldSpec("ttm_squeeze_pro", "TTM Squeeze Pro"),
    FieldSpec("ema_8", "EMA 8"),
    FieldSpec("ema_21", "EMA 21"),
    FieldSpec("atr_14", "ATR 14"),
]


# -----------------------------
# Table column “catalog”
# -----------------------------
TIME_COL_CANDIDATES = (
    "lower_date", "middle_date", "upper_date",
    "lower_timestamp", "middle_timestamp", "upper_timestamp",
)

STOCKS_BASE_COLS = [
    "symbol",
    "signal",
    # time cols get inserted dynamically right after signal
    "score_summary",
    "mtf_long_score",
    "mtf_short_score",

    # lower
    "lower_wyckoff_stage",
    "lower_exh_abs_pa_current_bar",
    "lower_exh_abs_pa_prior_bar",
    "lower_sig_vol_current_bar",
    "lower_sig_vol_prior_bar",
    "lower_spy_qqq_vol_ma_ratio",
    "lower_ma_trend_bullish",
    "lower_ma_trend_bearish",
    "lower_macdv_core_bull",
    "lower_macdv_core_bear",
    "lower_ttm_squeeze_pro",

    # middle
    "middle_wyckoff_stage",
    "middle_exh_abs_pa_prior_bar",
    "middle_sig_vol_current_bar",
    "middle_spy_qqq_vol_ma_ratio",

    # upper
    "upper_wyckoff_stage",
    "upper_exh_abs_pa_prior_bar",
]

STOCKS_OPTIONS_EXTRA_COLS = [
    "etf_symbol_primary",
    "etf_primary_long_score",
    "etf_primary_short_score",
    "etf_lower_primary_long_score",
    "etf_lower_primary_short_score",

    "etf_symbol_secondary",
    "etf_secondary_long_score",
    "etf_secondary_short_score",
    "etf_lower_secondary_long_score",
    "etf_lower_secondary_short_score",
]

FUTURES_BASE_COLS = [
    "symbol",
    "signal",
    # time cols inserted dynamically
    "score_summary",
    "mtf_long_score",
    "mtf_short_score",

    # lower
    "lower_wyckoff_stage",
    "lower_exh_abs_pa_current_bar",
    "lower_exh_abs_pa_prior_bar",
    "lower_sig_vol_current_bar",
    "lower_sig_vol_prior_bar",
    "lower_spy_qqq_vol_ma_ratio",
    "lower_ma_trend_bullish",
    "lower_ma_trend_bearish",
    "lower_macdv_core_bull",
    "lower_macdv_core_bear",
    "lower_ttm_squeeze_pro",

    # middle
    "middle_wyckoff_stage",
    "middle_exh_abs_pa_prior_bar",
    "middle_sig_vol_current_bar",
    "middle_spy_qqq_vol_ma_ratio",

    # upper
    "upper_wyckoff_stage",
    "upper_exh_abs_pa_prior_bar",
]
