from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml
import numpy as np
from datetime import datetime

from etl.window import parquet_path
from etl.universe import symbols_for_universe
from screens.etf_trend_engine import compute_etf_trend_scores


DATA = ROOT / "data"
CFG = ROOT / "config"
REF = ROOT / "ref"

MTF_CFG_PATH = CFG / "multi_timeframe_combos.yaml"

# Load multi-timeframe combos config
with open(MTF_CFG_PATH, "r") as f:
    MTF_CFG = yaml.safe_load(f)


# Columns we expect to be present after apply_core()
BASE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "sma_8",
    "ema_8",
    "ema_21",
    "wema_14",
    "ema_8_slope",
    "ema_21_slope",
    "volume_sma_20",
    "atr_14",
]



def attach_etf_trends_for_options_combo(
    combo_df: pd.DataFrame,
    combo_cfg: dict,
    timeframe_for_etf: str = "weekly",
) -> pd.DataFrame:
    """
    For options-eligible combos, join ETF trend scores using the
    symbol_to_etf_options_eligible.csv mapping and the ETF weekly scores.

    Assumes mapping has at least:
        - symbol
        - etf_symbol_primary
    """
    universe = combo_cfg.get("universe")
    if universe != "options_eligible":
        # Nothing to do for shortlist / futures / anything else
        return combo_df

    mapping_path = REF / "symbol_to_etf_options_eligible.csv"
    if not mapping_path.exists():
        print(f"[WARN] ETF mapping not found at {mapping_path}; skipping ETF guardrail join.")
        return combo_df

    mapping = pd.read_csv(mapping_path)

    if "symbol" not in mapping.columns or "etf_symbol_primary" not in mapping.columns:
        print(
            "[WARN] symbol_to_etf_options_eligible.csv missing "
            "'symbol' and/or 'etf_symbol_primary'; skipping ETF join."
        )
        return combo_df

    etf_scores = compute_etf_trend_scores(timeframe_for_etf)  # index=etf_symbol
    etf_scores = etf_scores.reset_index()  # columns: etf_symbol, etf_long_score, etf_short_score

    # 1) attach primary ETF symbol to combo rows
    combo = combo_df.merge(
        mapping[["symbol", "etf_symbol_primary", "etf_symbol_secondary"]],
        on="symbol",
        how="left",
    )

    
    # 2) attach ETF scores for that primary ETF
    combo = combo.merge(
        etf_scores.rename(columns={"etf_symbol": "etf_symbol_primary"}),
        on="etf_symbol_primary",
        how="left",
    )
    # Rename scores to make their role clear
    combo = combo.rename(
        columns={
            "etf_long_score": "etf_primary_long_score",
            "etf_short_score": "etf_primary_short_score",
        }
    )

    
    # 3) attach ETF scores for the ** secondary ** ETF
    combo = combo.merge(
        etf_scores.rename(columns={"etf_symbol": "etf_symbol_secondary"}),
        on="etf_symbol_secondary",
        how="left",
    )
    # Rename scores to make their role clear
    combo = combo.rename(
        columns={
            "etf_long_score": "etf_secondary_long_score",
            "etf_short_score": "etf_secondary_short_score",
        }
    )

    return combo


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
) -> tuple[str, str]:
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
            return "stocks_shortlist", "lower_exh_abs_pa_current_bar"
        # Family B: all other combos -> use prior-bar Exh/Abs on lower
        return "stocks_shortlist", "lower_exh_abs_pa_prior_bar"

    # Stocks: options-eligible universe
    if namespace == "stocks" and universe == "options_eligible":
        # For now we'll route options through the same evaluator,
        # but we might tune thresholds later.
        if pattern in ("daily-weekly-monthly", "weekly-monthly-quarterly"):
            return "stocks_options", "lower_exh_abs_pa_current_bar"
        if pattern == "monthly-quarterly-yearly":
            return "stocks_options", "lower_exh_abs_pa_prior_bar"
        # Default for any other pattern
        return "stocks_options", "lower_exh_abs_pa_prior_bar"

    # Futures: shortlist universe (stubs for later)
    if namespace == "futures" and universe == "shortlist_futures":
        # You can branch on pattern or combo_name later.
        return "futures_shortlist", "lower_exh_abs_pa_current_bar"

    # Fallback: no evaluator
    return "none", ""


def symbols_for_universe(universe: str) -> list[str]:
    """
    Map a universe name to a list of symbols, using CSV files in config/ or ref/.
    """
    if universe == "shortlist_stocks":
        p = CFG / "shortlist_stocks.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    if universe == "shortlist_futures":
        p = CFG / "shortlist_futures.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    if universe == "options_eligible":
        p = REF / "options_eligible.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    # Future: add ETF shortlist universes here if needed.
    return []


def _load_role_frame(
    namespace: str,
    timeframe: str,
    role: str,
) -> pd.DataFrame:
    """
    Load the snapshot for a given namespace+timeframe and reshape it into a
    role-prefixed frame keyed by symbol.

    Result:
        index: symbol
        columns: f"{role}_<original_col>" for all non-symbol columns
    """
    snap_path = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
    if not snap_path.exists():
        print(f"[WARN] Snapshot not found for {namespace} {timeframe}: {snap_path}")
        return pd.DataFrame()

    snap = pd.read_parquet(snap_path)
    if snap.empty:
        return pd.DataFrame()

    if "symbol" not in snap.columns:
        print(f"[WARN] Snapshot {snap_path} has no 'symbol' column; skipping.")
        return pd.DataFrame()

    # Use symbol as key; keep ALL other columns (prices + indicators)
    snap = snap.set_index("symbol")

    # Prefix everything with the role (lower_, middle_, upper_)
    snap = snap.add_prefix(f"{role}_")

    # Index is symbol, columns are role-prefixed
    return snap


def build_combo_df(
    namespace: str,
    combo_name: str,
    mtf_cfg: dict,
) -> pd.DataFrame:
    """
    Build a multi-timeframe combo dataframe:

        symbol,
        lower_* columns,
        middle_* columns,
        upper_* columns

    For each snapshot, we keep ALL columns (OHLCV + indicators) and simply
    prefix them with the role name. Joins are done on symbol.
    """
    ns_cfg = mtf_cfg.get(namespace, {})
    if combo_name not in ns_cfg:
        raise ValueError(f"Combo '{combo_name}' not found under namespace '{namespace}'")

    cfg = ns_cfg[combo_name]
    lower_tf = cfg["lower_tf"]
    middle_tf = cfg["middle_tf"]
    upper_tf = cfg["upper_tf"]

    lower = _load_role_frame(namespace, lower_tf, "lower")
    middle = _load_role_frame(namespace, middle_tf, "middle")
    upper = _load_role_frame(namespace, upper_tf, "upper")

    # Require at least lower + middle; upper can also be required if you prefer
    if lower.empty or middle.empty or upper.empty:
        print(f"[WARN] One or more role frames empty for combo '{combo_name}'.")
        return pd.DataFrame()

    # Inner join on symbol (index)
    combo = lower.join(middle, how="inner").join(upper, how="inner")

    if combo.empty:
        return combo

    # Bring symbol back as a column
    combo = combo.reset_index().rename(columns={"index": "symbol"})

    # 🔹 NEW: filter to the combo's universe symbols
    universe = cfg.get("universe")
    if universe:
        allowed = set(symbols_for_universe(universe))
        if allowed:
            combo = combo[combo["symbol"].isin(allowed)]

    return combo


def evaluate_stocks_shortlist_signal(
    row: pd.Series,
    exh_abs_col: str,
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

    # -----------------------------
    # Unpack fields
    # -----------------------------
    # Lower (timing / PA): usually daily
    lw_wyckoff = row.get("lower_wyckoff_stage", np.nan)
    lw_exh_abs = row.get(exh_abs_col, np.nan)  # current or prior bar, based on routing
    lw_sigvol = row.get("lower_significant_volume", np.nan)
    lw_vol_ratio = row.get("lower_spy_qqq_vol_ma_ratio", np.nan)
    lw_trend_cloud = row.get("lower_ma_trend_cloud", np.nan)
    lw_macdv = row.get("lower_macdv_core", np.nan)
    lw_sqz = row.get("lower_ttm_squeeze_pro", np.nan)

    # Middle: context / confirmation (e.g., weekly)
    md_wyckoff = row.get("middle_wyckoff_stage", np.nan) 
    md_exh_abs = row.get("middle_exh_abs_pa_prior_bar", np.nan) 
    md_sigvol = row.get("middle_significant_volume", np.nan)
    md_vol_ratio = row.get("middle_spy_qqq_vol_ma_ratio", np.nan)

    # Upper: regime (e.g., monthly)
    up_wyckoff = row.get("upper_wyckoff_stage", np.nan)
    up_exh_abs = row.get("upper_exh_abs_pa_prior_bar", np.nan) 

    long_score = 0.0
    short_score = 0.0

    # ======================================================
    # Block 1: Trend / Regime (upper + middle)
    # ======================================================
    # Upper regime bias: aligned bullish vs bearish
    if (up_wyckoff != np.nan and (up_wyckoff > 0 or up_exh_abs > 0)) or (up_wyckoff == np.nan and (md_wyckoff > 0 or md_exh_abs > 0)):
        long_score += 1.0
    if (up_wyckoff != np.nan and (up_wyckoff < 0 or up_exh_abs < 0)) or (up_wyckoff == np.nan and (md_wyckoff < 0 or md_exh_abs < 0)):
        short_score += 1.0

    # ======================================================
    # Block 2: Price Action / Momentum (lower)
    # ======================================================
    # Lower regime moving average trend cloud
    if lw_trend_cloud > 0:
        long_score += 1.0
    if lw_trend_cloud < 0:
        short_score += 1.0

    # Exh/Abs (current or prior bar, depending on combo family)
    if lw_exh_abs in (1.0, 2.0):
        long_score += 1.0
    if lw_exh_abs in (-1.0, -2.0):
        short_score += 1.0

    # MACDV Momentum with potential TTM Squeeze Pro Confirmation
    if lw_macdv == 2 or (lw_macdv == 1 and lw_sqz != np.nan and lw_sqz >= 0):
        long_score += 1.0
    if lw_macdv == -2 or (lw_macdv == -1 and lw_sqz != np.nan and lw_sqz <= 0):
        short_score += 1.0

    # ======================================================
    # Decision mapping (v1 thresholds, easy to tune)
    # ======================================================
    if long_score <= 0 and short_score <= 0:
        return "none", long_score, short_score

    if long_score >= 4.0:
        return "long", long_score, short_score

    if short_score >= 4.0:
        return "short", long_score, short_score

    return "none", long_score, short_score


def _aggregate_etf_score(row: pd.Series, cols: list[str]) -> float:
    """
    Combine primary / secondary ETF scores for one direction.

    Behavior:
      - If *both* columns are missing or NaN -> return np.nan (no ETF data)
      - Otherwise:
          - treat None / NaN as 0.0
          - return max(cleaned_scores)
    """
    raw_vals = []
    for c in cols:
        if c in row:
            raw_vals.append(row[c])
        else:
            raw_vals.append(np.nan)

    # If all values are missing/NaN, this symbol has no ETF mapping/data
    if all(pd.isna(v) for v in raw_vals):
        return np.nan

    cleaned = []
    for v in raw_vals:
        if v is None or pd.isna(v):
            cleaned.append(0.0)
        else:
            try:
                cleaned.append(float(v))
            except (TypeError, ValueError):
                cleaned.append(0.0)

    return max(cleaned)


def evaluate_stocks_options_signal(
    row: pd.Series,
    exh_abs_col: str,
) -> tuple[str, float, float]:
    """
    Options-eligible version of the equity signal.

    Strategy:
      - Reuse the shortlist scoring.
      - Then require significant volume on BOTH middle and lower timeframes
        for any 'long' or 'short' signal to stand.
      - Otherwise, downgrade to 'watch'.

    This keeps trend/PA logic identical, but enforces stronger participation
    for options trades.
    """
    base_signal, long_score, short_score = evaluate_stocks_shortlist_signal(row, exh_abs_col)

    vol_ratio_th1 = 0.10
    vol_ratio_th2 = 0.25

    # If there is no directional signal, nothing to add.
    if base_signal not in ("long", "short"):
        return base_signal, long_score, short_score

    up_wyckoff = row.get("upper_wyckoff_stage", np.nan)
    md_sigvol = row.get("middle_significant_volume", np.nan)
    md_vol_ratio = row.get("middle_spy_qqq_vol_ma_ratio", np.nan)
    lw_sigvol = row.get("lower_significant_volume", np.nan)
    lw_vol_ratio = row.get("lower_spy_qqq_vol_ma_ratio", np.nan)

    # ======================================================
    # Block 3: Volume / Participation (lower + middle)
    # ======================================================
    # Significant volume + beating SPY/QQQ volume baseline -> strong participation
    if up_wyckoff != np.nan and ((md_sigvol == 1.0 and md_vol_ratio > vol_ratio_th1) or (lw_sigvol == 1.0 and lw_vol_ratio > vol_ratio_th1)):
        long_score += 1.0
        short_score += 1.0
    if up_wyckoff == np.nan and ((md_sigvol == 1.0 and md_vol_ratio > vol_ratio_th2) or (lw_sigvol == 1.0 and lw_vol_ratio > vol_ratio_th2)):
        long_score += 1.0
        short_score += 1.0

    # ======================================================
    # Decision mapping (v1 thresholds, easy to tune)
    # ======================================================
    """
    if long_score <= 0 and short_score <= 0:
        return "none", long_score, short_score
    """
    
    if long_score >= 5.0:
        return "long", long_score, short_score

    if short_score >= 5.0:
        return "short", long_score, short_score

    if long_score < 5.0 and short_score < 5.0:
        return "none", long_score, short_score

    #return "none", long_score, short_score
    
    #ETF overlay: look at primary + secondary, but preserve "no data" as NaN
    etf_long = _aggregate_etf_score(
        row,
        ["etf_primary_long_score", "etf_secondary_long_score"],
    )
    etf_short = _aggregate_etf_score(
        row,
        ["etf_primary_short_score", "etf_secondary_short_score"],
    )

    # None / NaN -> treat as 0
    #etf_long = float(etf_long) if etf_long is not None else 0.0
    #etf_short = float(etf_short) if etf_short is not None else 0.0

    # Apply guardrails **only when ETF data exists**
    # Long side
    if not pd.isna(etf_long) and (base_signal == "long" or long_score >= 5) and etf_long < 4:
        base_signal = "watch"
    # Short side
    elif not pd.isna(etf_short) and (base_signal == "short" or short_score >= 5) and etf_short < 4:
        base_signal = "watch"

    #return signal, long_score, short_score


    # Otherwise keep the base directional signal
    return base_signal, long_score, short_score


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
    evaluator_name, exh_abs_col = _resolve_signal_routing(namespace, combo_name, combo_cfg)

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
    elif evaluator_name == "futures_shortlist":
        # TODO: implement evaluate_futures_shortlist_signal later
        return combo_df
    else:
        return combo_df

    signals: list[str] = []
    long_scores: list[float] = []
    short_scores: list[float] = []

    for _, row in combo_df.iterrows():
        sig, ls, ss = eval_fn(row, exh_abs_col)
        signals.append(sig)
        long_scores.append(ls)
        short_scores.append(ss)

    combo_df["signal"] = signals
    combo_df["mtf_long_score"] = long_scores
    combo_df["mtf_short_score"] = short_scores

    return combo_df


def run(namespace: str, combo_name: str):
    # Full multi-timeframe config (all combos for all namespaces)
    global MTF_CFG

    # Per-combo config (universe, lower_tf, middle_tf, upper_tf, etc.)
    combo_cfg = MTF_CFG[namespace][combo_name]

    # 1) Build combo rows (lower/middle/upper merged) using the FULL cfg
    combo_df = build_combo_df(namespace, combo_name, MTF_CFG)

    if combo_df is None or combo_df.empty:
        print(f"[INFO] No data for combo '{namespace}:{combo_name}'. Nothing to write.")
        return

    # 2) Apply multi-timeframe signal engine using the PER-COMBO cfg
    combo_df = basic_signal_logic(namespace, combo_name, combo_cfg, combo_df)

    # NEW: attach ETF guardrail info for options-eligible combos
    middle_tf = combo_cfg.get("middle_tf", "weekly") #<--- "weekly" here is just a default
    combo_df = attach_etf_trends_for_options_combo(
        combo_df,
        combo_cfg=combo_cfg,
        timeframe_for_etf=middle_tf,  # for DWM, this is 'weekly'
    )

    # 3) Save as before
    # Old convention: "combo_" + combo_name
    out = DATA / f"combo_{combo_name}.parquet"

    combo_df.to_parquet(out)
    print(f"[OK] Wrote combo snapshot to {out}")

    # ------------------------------------------------------------------
    # Also write a dated history snapshot for this combo run.
    # This gives us point-in-time scores (DWM, WMQ, intraday, futures, etc.)
    # without changing how the "current" file is used by the dashboard.
    # ------------------------------------------------------------------
    # Use UTC timestamp in a file-name-safe format (no ":" characters)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

    hist_dir = DATA / "combo_history" / namespace / combo_name
    hist_dir.mkdir(parents=True, exist_ok=True)

    hist_path = hist_dir / f"combo_{combo_name}_asof={ts}.parquet"
    combo_df.to_parquet(hist_path)
    print(f"[OK] Wrote combo history snapshot to {hist_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jobs/run_combo.py <namespace> <combo_name>")
        print("Example: python jobs/run_combo.py stocks stocks_c_dwm_shortlist")
        sys.exit(1)

    ns = sys.argv[1]
    combo = sys.argv[2]
    run(ns, combo)
