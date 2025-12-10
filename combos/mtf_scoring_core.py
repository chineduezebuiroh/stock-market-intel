from __future__ import annotations

# combos/mtf_scoring_core.py

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pathlib import Path

#from screens.etf_trend_engine import compute_etf_trend_scores
from etf.trend_engine import write_etf_trend_scores

from etf.guardrails import (
    attach_etf_trends_for_options_combo,
    aggregate_etf_score,
)

ROOT = Path(__file__).resolve().parents[1]
#REF = ROOT / "ref"
from core.paths import REF #, CFG  # NEW


@dataclass
class MTFScore:
    side: str          # "long", "short", "none"
    long_score: float
    short_score: float


class MTFScorer(Protocol):
    def __call__(self, row: pd.Series) -> MTFScore: ...
    

# ==============================================================================
# Resolve signal-routing logic
# ==============================================================================
def basic_signal_logic(
    namespace: str,
    combo_name: str,
    combo_cfg: dict,
    combo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach multi-timeframe signal columns to combo_df based on:
      - namespace (stocks / futures)
      - universe (shortlist / options / futures shortlist)
      - timeframe pattern (dwm / wmq / mqy, etc.)

    Uses a small routing helper to pick:
      - evaluator_name: which evaluation family to use
      - exh_abs_col: which lower_* Exh/Abs PA column to reference
    """
    evaluator_name, exh_abs_col, sig_vol_col = _resolve_signal_routing(namespace, combo_name, combo_cfg)

    combo_df = combo_df.copy()
    combo_df["signal"] = "none"
    combo_df["mtf_long_score"] = 0.0
    combo_df["mtf_short_score"] = 0.0

    if evaluator_name == "none" or not exh_abs_col:
        # No configured evaluator for this combo; leave neutral
        return combo_df

    if evaluator_name == "stocks_shortlist":
        eval_fn = evaluate_stocks_shortlist_signal
    elif evaluator_name == "stocks_options":
        eval_fn = evaluate_stocks_options_signal
    elif evaluator_name == "futures_base":
        eval_fn = score_futures_base_signal
    elif evaluator_name == "futures_4hdw":
        eval_fn = score_futures_4hdw_signal
    elif evaluator_name == "futures_dwm":
        eval_fn = score_futures_dwm_signal
    else:
        # No evaluator: leave neutral
        return combo_df
    

    signals: list[str] = []
    long_scores: list[float] = []
    short_scores: list[float] = []

    for _, row in combo_df.iterrows():
        sig, ls, ss = eval_fn(row, exh_abs_col, sig_vol_col)
        signals.append(sig)
        long_scores.append(ls)
        short_scores.append(ss)

    combo_df["signal"] = signals
    combo_df["signal_side"] = signals # <--- clean up later?
    combo_df["mtf_long_score"] = long_scores
    combo_df["mtf_short_score"] = short_scores

    return combo_df


def _tf_pattern(combo_cfg: dict) -> str:
    """
    Convenience helper: turn lower/middle/upper into a simple pattern string
    like 'daily-weekly-monthly', 'weekly-monthly-quarterly', etc.
    """
    lower_tf = combo_cfg.get("lower_tf", "")
    middle_tf = combo_cfg.get("middle_tf", "")
    upper_tf = combo_cfg.get("upper_tf", "")
    return f"{lower_tf}-{middle_tf}-{upper_tf}"


def _resolve_signal_routing(
    namespace: str,
    combo_name: str,
    combo_cfg: dict,
) -> tuple[str, str, str]:
    """
    Decide which evaluator to use and which lower_* Exh/Abs column to read.

    Returns:
        evaluator_name: str (e.g. 'stocks_shortlist', 'stocks_options', 'futures_shortlist', 'none')
        exh_abs_col:    str (e.g. 'lower_exh_abs_pa_current_bar' or 'lower_exh_abs_pa_prior_bar')
    """
    universe = combo_cfg.get("universe", "")
    pattern = _tf_pattern(combo_cfg)

    # Stocks: shortlist universe
    if namespace == "stocks" and universe == "shortlist_stocks":
        # Family A: DWM & WMQ -> use current-bar Exh/Abs on *lower* timeframe
        if pattern in ("daily-weekly-monthly", "weekly-monthly-quarterly"):
            return "stocks_shortlist", "lower_exh_abs_pa_current_bar", "lower_sig_vol_current_bar"
        # Family B: all other combos -> use prior-bar Exh/Abs on lower
        return "stocks_shortlist", "lower_exh_abs_pa_prior_bar", "lower_sig_vol_prior_bar"

    # Stocks: options-eligible universe
    if namespace == "stocks" and universe == "options_eligible":
        # For now we'll route options through the same evaluator,
        # but we might tune thresholds later.
        if pattern in ("daily-weekly-monthly", "weekly-monthly-quarterly"):
            return "stocks_options", "lower_exh_abs_pa_current_bar", "lower_sig_vol_current_bar"
        if pattern == "monthly-quarterly-yearly":
            return "stocks_options", "lower_exh_abs_pa_prior_bar", "lower_sig_vol_prior_bar"
        # Default for any other pattern
        return "stocks_options", "lower_exh_abs_pa_prior_bar", "lower_sig_vol_prior_bar"

    # Futures: shortlist universe (stubs for later)
    if namespace == "futures": #and universe == "shortlist_futures":
        # 1h / 4h / D
        if pattern == "intraday_1h-intraday_4h-daily":
            return "futures_base", "lower_exh_abs_pa_prior_bar", "lower_sig_vol_prior_bar"
        # 4h / D / W
        if pattern == "intraday_4h-daily-weekly":
            return "futures_4hdw", "lower_exh_abs_pa_prior_bar", "lower_sig_vol_prior_bar"
        # D / W / M
        if pattern == "daily-weekly-monthly":
            return "futures_dwm", "lower_exh_abs_pa_prior_bar", "lower_sig_vol_prior_bar"

        # Fallback: no evaluator
        return "none", "", ""

# ===========================================================================
# STOCK (SHORTLIST) scoring functions
# ===========================================================================
def evaluate_stocks_shortlist_signal(
    row: pd.Series,
    exh_abs_col: str,
    sig_vol_col: str,
) -> tuple[str, float, float]:
    """
    Multi-timeframe evaluation for STOCKS in the shortlist universe
    (and optionally reused for options as a starting point).

    Functional grouping:
      - Block 1: Trend / Regime (upper + middle)
      - Block 2: Volume / Participation (lower + benchmarks)
      - Block 3: Price Action / Momentum (mostly lower)

    Returns:
        signal: "long" | "short" | "watch" | "none"
        long_score: float
        short_score: float
    """

    # -----------------------------------------------------
    # Unpack fields
    # -----------------------------------------------------

    # Lower (timing / PA): usually daily
    lw_wyckoff = row.get("lower_wyckoff_stage", np.nan)
    lw_exh_abs = row.get(exh_abs_col, np.nan)  # current or prior bar, based on routing
    lw_sigvol = row.get(sig_vol_col, np.nan)
    lw_vol_ratio = row.get("lower_spy_qqq_vol_ma_ratio", np.nan)
    lw_ma_trend_bull = row.get("lower_ma_trend_bullish", np.nan)
    lw_ma_trend_bear = row.get("lower_ma_trend_bearish", np.nan)
    lw_macdv = row.get("lower_macdv_core", np.nan)
    lw_sqz = row.get("lower_ttm_squeeze_pro", np.nan)

    # Middle: context / confirmation (e.g., weekly)
    md_wyckoff = row.get("middle_wyckoff_stage", np.nan) 
    md_exh_abs = row.get("middle_exh_abs_pa_prior_bar", np.nan) 
    md_sigvol = row.get("middle_sig_vol_current_bar", np.nan)
    md_vol_ratio = row.get("middle_spy_qqq_vol_ma_ratio", np.nan)
    
    # Upper: regime (e.g., monthly)
    up_wyckoff = row.get("upper_wyckoff_stage", np.nan)
    up_exh_abs = row.get("upper_exh_abs_pa_prior_bar", np.nan) 
    
    long_score = 0.0
    short_score = 0.0

    # ------------------------------------------------------
    # Block 1: Trend / Regime (upper + middle)
    # ------------------------------------------------------
    # Upper regime bias: aligned bullish vs bearish
    if (~np.isnan(up_wyckoff) and (up_wyckoff > 0 or up_exh_abs > 0)) or (np.isnan(up_wyckoff) and (md_wyckoff > 0 or md_exh_abs > 0)):
        long_score += 1.0
    if (~np.isnan(up_wyckoff) and (up_wyckoff < 0 or up_exh_abs < 0)) or (np.isnan(up_wyckoff) and (md_wyckoff < 0 or md_exh_abs < 0)):
        short_score += 1.0

    # ------------------------------------------------------
    # Block 2: Price Action / Momentum (lower)
    # ------------------------------------------------------
    # Lower regime moving average trend cloud
    if lw_ma_trend_bull > 0:
        long_score += 1.0
    if lw_ma_trend_bear < 0:
        short_score += 1.0

    # Exh/Abs (current or prior bar, depending on combo family)
    if lw_exh_abs in (1.0, 2.0):
        long_score += 1.0
    if lw_exh_abs in (-1.0, -2.0):
        short_score += 1.0

    # MACDV Momentum with potential TTM Squeeze Pro Confirmation
    if lw_macdv == 2 or (lw_macdv == 1 and ~np.isnan(lw_sqz) and lw_sqz >= 0):
        long_score += 1.0
    if lw_macdv == -2 or (lw_macdv == -1 and ~np.isnan(lw_sqz) and lw_sqz <= 0):
        short_score += 1.0

    # ------------------------------------------------------
    # Decision mapping (v1 thresholds, easy to tune)
    # ------------------------------------------------------
    if long_score <= 0 and short_score <= 0:
        return "none", long_score, short_score

    if long_score >= 4.0:
        return "long", long_score, short_score

    if short_score >= 4.0:
        return "short", long_score, short_score

    return "none", long_score, short_score



# =========================================================================
# STOCK (OPTIONS-ELIGIBLE) scoring functions
# =========================================================================
"""
def evaluate_stocks_options_signal(
    row: pd.Series,
    exh_abs_col: str,
    sig_vol_col: str,
) -> tuple[str, float, float]:
"""
"""
    Options-eligible version of the equity signal.

    Strategy:
      - Reuse the shortlist scoring.
      - Then require significant volume on EITHER middle or lower timeframes
        for any 'long' or 'short' signal to stand.
      - Otherwise, downgrade to 'watch'.

    This keeps trend/PA logic identical, but enforces stronger participation
    for options trades.
"""
"""
    base_signal, long_score, short_score = evaluate_stocks_shortlist_signal(row, exh_abs_col, sig_vol_col)

    vol_ratio_th1 = 0.10
    vol_ratio_th2 = 0.25

    # If there is no directional signal, nothing to add.
    if base_signal not in ("long", "short"):
        return base_signal, long_score, short_score

    up_wyckoff = row.get("upper_wyckoff_stage", np.nan)
    md_sigvol = row.get("middle_sig_vol_current_bar", np.nan)
    md_vol_ratio = row.get("middle_spy_qqq_vol_ma_ratio", np.nan)
    lw_sigvol = row.get(sig_vol_col, np.nan)
    lw_vol_ratio = row.get("lower_spy_qqq_vol_ma_ratio", np.nan)

    # ------------------------------------------------------
    # Block 3: Volume / Participation (lower + middle)
    # ------------------------------------------------------
    # Significant volume + beating SPY/QQQ volume baseline -> strong participation
    if ~np.isnan(up_wyckoff) and ((md_sigvol == 1.0 and md_vol_ratio > vol_ratio_th1) or (lw_sigvol == 1.0 and lw_vol_ratio > vol_ratio_th1)):
        long_score += 1.0
        short_score += 1.0
    if np.isnan(up_wyckoff) and ((md_sigvol == 1.0 and md_vol_ratio > vol_ratio_th2) or (lw_sigvol == 1.0 and lw_vol_ratio > vol_ratio_th2)):
        long_score += 1.0
        short_score += 1.0

    # ------------------------------------------------------
    # Decision mapping (v1 thresholds, easy to tune)
    # ------------------------------------------------------
    
    if base_signal == "long" and long_score < 5.0:
        base_signal = "none"

    if base_signal == "short" and short_score < 5.0:
        base_signal = "none"
    
    #ETF overlay: look at primary + secondary, but preserve "no data" as NaN
    etf_long = aggregate_etf_score(
        row,
        ["etf_primary_long_score", "etf_secondary_long_score"],
    )
    etf_short = aggregate_etf_score(
        row,
        ["etf_primary_short_score", "etf_secondary_short_score"],
    )

    # Apply guardrails **only when ETF data exists**
    # Long side
    if base_signal == "long" and not pd.isna(etf_long) and etf_long < 4:
        base_signal = "watch"
    # Short side
    elif base_signal == "short" and not pd.isna(etf_short) and etf_short < 4:
        base_signal = "watch"

    return base_signal, long_score, short_score
"""


def evaluate_stocks_options_signal(
    row: pd.Series,
    exh_abs_col: str,
) -> tuple[str, float, float]:
    """
    Options-eligible version of the equity signal.

    Strategy:
      - Reuse the shortlist scoring.
      - Then require stronger volume / participation for the signal to stand.
      - Finally, apply ETF trend guardrails:
          * downgrade LONG to WATCH if ETF long score < 4
          * downgrade SHORT to WATCH if ETF short score < 4
    """
    # 1) Base equity MTF logic (no ETF guardrails yet)
    base_signal, long_score, short_score = evaluate_stocks_shortlist_signal(row, exh_abs_col, sig_vol_col)

    # If there's no directional bias, nothing more to do.
    if base_signal not in ("long", "short"):
        return base_signal, long_score, short_score

    # 2) Volume / participation overlay
    vol_ratio_th1 = 0.10
    vol_ratio_th2 = 0.25

    up_wyckoff = row.get("upper_wyckoff_stage", np.nan)
    md_sigvol = row.get("middle_significant_volume", np.nan)
    md_vol_ratio = row.get("middle_spy_qqq_vol_ma_ratio", np.nan)
    lw_sigvol = row.get(sig_vol_col, np.nan)
    lw_vol_ratio = row.get("lower_spy_qqq_vol_ma_ratio", np.nan)

    # Strong participation when weekly regime is clear (upper Wyckoff non-NaN)
    if not pd.isna(up_wyckoff) and (
        (md_sigvol == 1.0 and md_vol_ratio > vol_ratio_th1)
        or (lw_sigvol == 1.0 and lw_vol_ratio > vol_ratio_th1)
    ):
        long_score += 1.0
        short_score += 1.0

    # If upper regime is ambiguous, require stronger volume beat
    if pd.isna(up_wyckoff) and (
        (md_sigvol == 1.0 and md_vol_ratio > vol_ratio_th2)
        or (lw_sigvol == 1.0 and lw_vol_ratio > vol_ratio_th2)
    ):
        long_score += 1.0
        short_score += 1.0

    # 3) Tighten thresholds for options
    #    (this was where the == vs = bug used to live)
    if base_signal == "long" and long_score < 5.0:
        base_signal = "none"

    if base_signal == "short" and short_score < 5.0:
        base_signal = "none"

    # If the tightened thresholds killed the signal, stop here.
    if base_signal not in ("long", "short"):
        return base_signal, long_score, short_score

    # 4) ETF overlay: guardrails using primary + secondary scores
    etf_long = aggregate_etf_score(
        row,
        ["etf_primary_long_score", "etf_secondary_long_score"],
    )
    etf_short = aggregate_etf_score(
        row,
        ["etf_primary_short_score", "etf_secondary_short_score"],
    )

    # Apply guardrails **only when ETF data exists** (etf_* is not NaN).
    # Long side
    if base_signal == "long" and not pd.isna(etf_long) and etf_long < 4.0:
        base_signal = "watch"

    # Short side
    if base_signal == "short" and not pd.isna(etf_short) and etf_short < 4.0:
        base_signal = "watch"

    return base_signal, long_score, short_score


# ========================================================================
# FUTURES scoring functions – one per combo
# ========================================================================

def score_futures_base_signal(
    row: pd.Series,
    exh_abs_col: str,
    sig_vol_col: str,
) -> tuple[str, float, float]:
    
    """
    Futures combo 1: 1h / 4h / Daily
    Treat:
      lower  = 1h
      middle = 4h
      upper  = daily
    but with a slightly more intraday-leaning weighting.
    """

    # -----------------------------------------------------
    # Unpack fields
    # -----------------------------------------------------

    # Lower (timing / PA): usually daily
    lw_wyckoff = row.get("lower_wyckoff_stage", np.nan)
    lw_exh_abs = row.get(exh_abs_col, np.nan)  # current or prior bar, based on routing
    lw_sigvol = row.get(sig_vol_col, np.nan)
    lw_vol_ratio = row.get("lower_spy_qqq_vol_ma_ratio", np.nan)
    lw_ma_trend_bull = row.get("lower_ma_trend_bullish", np.nan)
    lw_ma_trend_bear = row.get("lower_ma_trend_bearish", np.nan)
    lw_macdv = row.get("lower_macdv_core", np.nan)
    lw_sqz = row.get("lower_ttm_squeeze_pro", np.nan)

    # Middle: context / confirmation (e.g., weekly)
    md_wyckoff = row.get("middle_wyckoff_stage", np.nan) 
    md_exh_abs = row.get("middle_exh_abs_pa_prior_bar", np.nan) 
    md_sigvol = row.get("middle_sig_vol_current_bar", np.nan)
    md_vol_ratio = row.get("middle_spy_qqq_vol_ma_ratio", np.nan)
    
    # Upper: regime (e.g., monthly)
    up_wyckoff = row.get("upper_wyckoff_stage", np.nan)
    up_exh_abs = row.get("upper_exh_abs_pa_prior_bar", np.nan) 
    
    long_score = 0.0
    short_score = 0.0

    # -----------------------------------------------------
    # Block 1: Trend / Regime (upper + middle)
    # -----------------------------------------------------
    # Upper regime bias: aligned bullish vs bearish
    if (~np.isnan(up_wyckoff) and (up_wyckoff > 0 or up_exh_abs > 0)) or (np.isnan(up_wyckoff) and (md_wyckoff > 0 or md_exh_abs > 0)):
        long_score += 1.0
    if (~np.isnan(up_wyckoff) and (up_wyckoff < 0 or up_exh_abs < 0)) or (np.isnan(up_wyckoff) and (md_wyckoff < 0 or md_exh_abs < 0)):
        short_score += 1.0

    # ------------------------------------------------------
    # Block 2: Price Action / Momentum (lower)
    # ------------------------------------------------------
    # Lower regime moving average trend cloud
    if lw_ma_trend_bull > 0:
        long_score += 1.0
    if lw_ma_trend_bear < 0:
        short_score += 1.0

    # Exh/Abs (current or prior bar, depending on combo family)
    if lw_exh_abs in (1.0, 2.0):
        long_score += 1.0
    if lw_exh_abs in (-1.0, -2.0):
        short_score += 1.0

    # MACDV Momentum with potential TTM Squeeze Pro Confirmation
    if lw_macdv == 2 or (lw_macdv == 1 and ~np.isnan(lw_sqz) and lw_sqz >= 0):
        long_score += 1.0
    if lw_macdv == -2 or (lw_macdv == -1 and ~np.isnan(lw_sqz) and lw_sqz <= 0):
        short_score += 1.0

    # ------------------------------------------------------
    # Decision mapping (v1 thresholds, easy to tune)
    # ------------------------------------------------------
    if long_score <= 0 and short_score <= 0:
        return "none", long_score, short_score

    if long_score >= 4.0:
        return "long", long_score, short_score

    if short_score >= 4.0:
        return "short", long_score, short_score

    return "none", long_score, short_score


def score_futures_4hdw_signal(
    row: pd.Series,
    exh_abs_col: str,
    sig_vol_col: str,
) -> tuple[str, float, float]:
    """
    Futures combo 2: 4h / Daily / Weekly
    lower  = 4h
    middle = daily
    upper  = weekly

    Strategy:
      - Reuse the futures base signal scoring.
      - Then require significant volume on middle timeframe
        for any 'long' or 'short' signal to stand.
    """

    base_signal, long_score, short_score = score_futures_base_signal(row, exh_abs_col, sig_vol_col)

    # If there is no directional signal, nothing to add.
    if base_signal not in ("long", "short"):
        return base_signal, long_score, short_score

    up_wyckoff = row.get("upper_wyckoff_stage", np.nan)
    md_sigvol = row.get("middle_sig_vol_current_bar", np.nan)
    lw_sigvol = row.get(sig_vol_col, np.nan)

    # ------------------------------------------------------
    # Block 3: Volume / Participation (lower + middle)
    # ------------------------------------------------------
    # Significant volume -> strong participation
    if ~np.isnan(up_wyckoff) and md_sigvol == 1.0:
        long_score += 1.0
        short_score += 1.0

    # ------------------------------------------------------
    # Decision mapping (v1 thresholds, easy to tune)
    # ------------------------------------------------------
    if base_signal == "long" and long_score < 5.0:
        base_signal = "none"

    if base_signal == "short" and short_score < 5.0:
        base_signal = "none"

    return base_signal, long_score, short_score



def score_futures_dwm_signal(
    row: pd.Series,
    exh_abs_col: str,
    sig_vol_col: str,
) -> tuple[str, float, float]:
    """
    Futures combo 2: Daily / Weekly / Monthly
    lower  = daily
    middle = weekly
    upper  = monthly

    Strategy:
      - Reuse the futures base signal scoring.
      - Then require significant volume on middle or lower
        timeframe for any 'long' or 'short' signal to stand.
    """

    base_signal, long_score, short_score = score_futures_base_signal(row, exh_abs_col, sig_vol_col)

    # If there is no directional signal, nothing to add.
    if base_signal not in ("long", "short"):
        return base_signal, long_score, short_score

    up_wyckoff = row.get("upper_wyckoff_stage", np.nan)
    md_sigvol = row.get("middle_sig_vol_current_bar", np.nan)
    lw_sigvol = row.get(sig_vol_col, np.nan)

    # ------------------------------------------------------
    # Block 3: Volume / Participation (lower + middle)
    # ------------------------------------------------------
    # Significant volume -> strong participation
    if ~np.isnan(up_wyckoff) and (md_sigvol == 1.0 or lw_sigvol == 1.0):
        long_score += 1.0
        short_score += 1.0

    # ------------------------------------------------------
    # Decision mapping (v1 thresholds, easy to tune)
    # ------------------------------------------------------
    if base_signal == "long" and long_score < 5.0:
        base_signal = "none"

    if base_signal == "short" and short_score < 5.0:
        base_signal = "none"

    return base_signal, long_score, short_score
