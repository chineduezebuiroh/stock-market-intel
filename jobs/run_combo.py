import sys
from pathlib import Path
import pandas as pd
import yaml
import numpy as np


# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from etl.window import parquet_path

DATA = ROOT / "data"
CFG = ROOT / "config"

MTF_CFG_PATH = CFG / "multi_timeframe_combos.yaml"

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


def load_multi_tf_config() -> dict:
    """Load the multi-timeframe combos config."""
    path = CFG / "multi_timeframe_combos.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
        p = ROOT / "ref" / "options_eligible.csv"
        return pd.read_csv(p)["symbol"].tolist() if p.exists() else []

    # Future: add ETF shortlist universes here if needed.
    return []


def load_role_frame(namespace: str, timeframe: str, symbols, role_prefix: str) -> pd.DataFrame:
    """
    Load the *snapshot* for namespace+timeframe (e.g. snapshot_stocks_daily.parquet),
    filter to the given symbols, and prefix columns with role_prefix:

        lower_open, lower_high, ..., lower_atr_14

    One row per symbol, flat column names.
    """
    snap_path = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
    if not snap_path.exists():
        print(f"[WARN] Snapshot not found for {namespace} {timeframe}: {snap_path}")
        return pd.DataFrame()

    df = pd.read_parquet(snap_path)
    if df.empty:
        print(f"[WARN] Snapshot {snap_path} is empty.")
        return pd.DataFrame()

    if "symbol" not in df.columns:
        print(f"[WARN] Snapshot {snap_path} has no 'symbol' column; cannot build combo frame.")
        return pd.DataFrame()

    # Filter to our universe + index by symbol
    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        print(f"[WARN] No overlapping symbols for snapshot {snap_path} and universe.")
        return pd.DataFrame()

    df = df.set_index("symbol")

    # Use only the BASE_COLS that actually exist in this snapshot
    cols = [c for c in BASE_COLS if c in df.columns]
    if not cols:
        print(f"[WARN] No BASE_COLS found in snapshot {snap_path}.")
        return pd.DataFrame()

    sub = df[cols].rename(columns={c: f"{role_prefix}{c}" for c in cols})
    sub.columns = sub.columns.astype(str)
    return sub


def build_combo_df(namespace: str, combo_name: str, cfg: dict) -> pd.DataFrame:
    """
    For a given combo, load lower/middle/upper latest bars and join them by symbol.

    Result:
      - index: symbol
      - columns: lower_*, middle_*, upper_* (flat strings)
    """
    ns_cfg = cfg.get(namespace, {})
    combo_cfg = ns_cfg.get(combo_name)
    if combo_cfg is None:
        raise ValueError(f"Combo '{combo_name}' not found under namespace '{namespace}'")

    lower_tf = combo_cfg["lower_tf"]
    middle_tf = combo_cfg.get("middle_tf")
    upper_tf = combo_cfg["upper_tf"]
    universe = combo_cfg["universe"]

    symbols = symbols_for_universe(universe)
    if not symbols:
        print(f"[WARN] No symbols for universe '{universe}'.")
        return pd.DataFrame()

    lower_df = load_role_frame(namespace, lower_tf, symbols, "lower_")
    if lower_df.empty:
        print(f"[WARN] No lower timeframe data for combo '{combo_name}'.")
        return pd.DataFrame()

    frames = [lower_df]

    if middle_tf and str(middle_tf).lower() != "null":
        middle_df = load_role_frame(namespace, middle_tf, symbols, "middle_")
        if middle_df.empty:
            print(f"[WARN] No middle timeframe data for combo '{combo_name}'.")
        else:
            frames.append(middle_df)

    upper_df = load_role_frame(namespace, upper_tf, symbols, "upper_")
    if upper_df.empty:
        print(f"[WARN] No upper timeframe data for combo '{combo_name}'.")
        return pd.DataFrame()
    frames.append(upper_df)

    combo_df = frames[0]
    for f in frames[1:]:
        combo_df = combo_df.join(f, how="inner")

    if combo_df.empty:
        print(f"[WARN] Joined combo dataframe for '{combo_name}' is empty.")
        return combo_df

    combo_df.columns = combo_df.columns.astype(str)
    return combo_df


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
    md_wyckoff = row.get("middle_wyckoff_stage", np.nan)  # if you wire it later
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
    if (up_wyckoff != np.nan and (up_wyckoff > 0 or up_exh_abs > 0)
            ) or 
        (up_wyckoff == np.nan and (md_wyckoff > 0 or md_exh_abs > 0)
            ):
            long_score += 1.0
    if (up_wyckoff != np.nan and (up_wyckoff < 0 or up_exh_abs < 0)
            ) or 
        (up_wyckoff == np.nan and (md_wyckoff < 0 or md_exh_abs < 0)
            ):
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
    if lw_macdv == 2 or 
            (lw_macdv == 1 and lw_sqz != np.nan and lw_sqz >= 0):
                long_score += 1.0
    if lw_macdv == -2 or 
            (lw_macdv == -1 and lw_sqz != np.nan and lw_sqz <= 0):
                short_score += 1.0

    # ======================================================
    # Block 3: Volume / Participation (lower + benchmark)
    # ======================================================
    """
    # Significant volume + beating SPY/QQQ volume baseline -> strong participation
    if lw_sigvol == 1.0 and lw_vol_ratio >= 1.0:
        long_score += 1.0
    # For now, keep shorts less volume-driven; you can add a bearish pattern later.
    """

    # ======================================================
    # Decision mapping (v1 thresholds, easy to tune)
    # ======================================================
    if long_score <= 0 and short_score <= 0:
        return "none", long_score, short_score

    if long_score >= 4.0:
        return "long", long_score, short_score

    if short_score >= 4.0:
        return "short", long_score, short_score

    return "watch", long_score, short_score

"""
def basic_signal_logic(combo_df: pd.DataFrame) -> pd.DataFrame:
    
    #First-pass placeholder MTF logic.
    #Right now, just tags everything as 'watch'.
    #We'll replace this with your real EMA/SMA/Wilder logic later.
    
    df = combo_df.copy()
    df["signal"] = "watch"
    return df
"""

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

    if evaluator_name in ("stocks_shortlist", "stocks_options"):
        eval_fn = evaluate_stocks_shortlist_signal
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


"""
def run_combo(namespace: str, combo_name: str):
    with open(MTF_CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    df = build_combo_df(namespace, combo_name, cfg)
    if df.empty:
        print(f"[INFO] No data for combo '{namespace}:{combo_name}'. Nothing to write.")
        return

    out = basic_signal_logic(df)
    out = out.reset_index()  # restore `symbol` column

    # ---- FIX REDUNDANT FILENAME ----
    clean_name = combo_name
    prefix = f"{namespace}_"
    if combo_name.startswith(prefix):
        clean_name = combo_name[len(prefix):]

    out_path = DATA / f"combo_{namespace}_{clean_name}.parquet"
    out.to_parquet(out_path)
    print(f"[OK] Wrote combo snapshot to {out_path}")
"""


def run(namespace: str, combo_name: str):
    ns_cfg = MTF_CFG[namespace][combo_name]

    combo_df = build_combo_df(namespace, combo_name, ns_cfg)

    if combo_df is None or combo_df.empty:
        print(f"[INFO] No data for combo '{namespace}:{combo_name}'. Nothing to write.")
        return

    # --- NEW: apply multi-timeframe signal engine ---
    combo_df = basic_signal_logic(namespace, combo_name, ns_cfg, combo_df)

    out = DATA / f"combo_{namespace}_{combo_name}.parquet"
    combo_df.to_parquet(out)
    print(f"[OK] Wrote combo snapshot to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jobs/run_combo.py <namespace> <combo_name>")
        print("Example: python jobs/run_combo.py stocks stocks_c_dwm_shortlist")
        sys.exit(1)

    ns = sys.argv[1]
    combo = sys.argv[2]
    run(ns, combo)
