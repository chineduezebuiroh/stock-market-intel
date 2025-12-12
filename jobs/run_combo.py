from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import DATA, CFG, REF
from core import storage

import pandas as pd
import yaml
import numpy as np
from datetime import datetime

#from etl.window import parquet_path
#from etl.universe import symbols_for_universe
#from screens.etf_trend_engine import compute_etf_trend_scores
from combos.mtf_scoring_core import basic_signal_logic
from etf.guardrails import attach_etf_trends_for_options_combo


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


def _load_role_frame(namespace: str, timeframe: str, role: str) -> pd.DataFrame:
    """
    Load the snapshot for a given namespace+timeframe and reshape it into a
    role-prefixed frame keyed by symbol.

    Result:
        index: symbol
        columns: f"{role}_<original_col>" for all non-symbol columns

    NEW:
        - Preserve a per-role timestamp column (e.g. lower_date / middle_date)
          derived from either the index or an explicit 'date'/'timestamp' col.
    """
    snap_path = DATA / f"snapshot_{namespace}_{timeframe}.parquet"
    """if not snap_path.exists():"""
    if not storage.exists(snap_path):
        print(f"[WARN] Snapshot not found for {namespace} {timeframe}: {snap_path}")
        return pd.DataFrame()

    """snap = pd.read_parquet(snap_path)"""
    snap = storage.load_parquet(snap_path)
    if snap.empty:
        return pd.DataFrame()

    # -----------------------------
    # Detect / normalize time column
    # -----------------------------
    time_col = None

    if isinstance(snap.index, pd.DatetimeIndex):
        # Use existing index name if present, otherwise call it 'date'
        time_col = snap.index.name or "date"
        snap = snap.reset_index()  # bring time into a column
    else:
        # Try common column names
        for candidate in ("date", "timestamp", "datetime"):
            if candidate in snap.columns:
                time_col = candidate
                break

    if "symbol" not in snap.columns:
        print(f"[WARN] Snapshot {snap_path} has no 'symbol' column; skipping.")
        return pd.DataFrame()

    # Reorder columns to: symbol, time_col (if any), then rest
    other_cols = [c for c in snap.columns if c not in ("symbol", time_col)]
    ordered_cols = ["symbol"] + ([time_col] if time_col else []) + other_cols
    snap = snap[ordered_cols]

    # Use symbol as key; keep ALL other columns (prices + indicators + time)
    snap = snap.set_index("symbol")

    # Prefix everything with the role (lower_, middle_, upper_)
    snap = snap.add_prefix(f"{role}_")

    # Index is symbol, columns are role-prefixed (e.g. lower_date, middle_date)
    return snap


def build_combo_df(namespace: str, combo_name: str, mtf_cfg: dict) -> pd.DataFrame:
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

    # 🔍 DEBUG: see what CI is actually loading
    print(
        f"[DEBUG] {combo_name} frames: "
        f"lower={lower.shape if not lower.empty else 'EMPTY'}, "
        f"middle={middle.shape if not middle.empty else 'EMPTY'}, "
        f"upper={upper.shape if not upper.empty else 'EMPTY'}"
    )

    # Require at least lower + middle + upper
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


def run(namespace: str, combo_name: str):
    # Full multi-timeframe config (all combos for all namespaces)
    global MTF_CFG

    # Per-combo config (universe, lower_tf, middle_tf, upper_tf, etc.)
    combo_cfg = MTF_CFG[namespace][combo_name]

    # 1) Build combo rows (lower/middle/upper merged) using the FULL cfg
    combo_df = build_combo_df(namespace, combo_name, MTF_CFG)

    if combo_df is None or combo_df.empty:
        """
        print(f"[INFO] No data for combo '{namespace}:{combo_name}'. Nothing to write.")
        return
        """
        raise RuntimeError(
            f"[FATAL] Combo '{namespace}:{combo_name}' produced no rows. "
            f"Check snapshots and upstream data."
        )

    # 2) Apply multi-timeframe signal engine using the PER-COMBO cfg
    # NEW: attach ETF guardrail info for options-eligible combos
    lower_tf = combo_cfg.get("lower_tf")
    middle_tf = combo_cfg.get("middle_tf", "weekly") #<--- "weekly" here is just a default

    # 🔹 Attach ETF scores FIRST (so scoring can see them)
    combo_df = attach_etf_trends_for_options_combo(
        combo_df,
        combo_cfg=combo_cfg,
        lower_tf=lower_tf,
        middle_tf=middle_tf,
    )
    
    combo_df = basic_signal_logic(namespace, combo_name, combo_cfg, combo_df)

    # 3) Save current combo snapshot
    out = DATA / f"combo_{combo_name}.parquet"
    """combo_df.to_parquet(out)"""
    storage.save_parquet(combo_df, out)
    print(f"[OK] Wrote combo snapshot to {out}")   

    # ------------------------------------------------------------------
    # Also write a dated history snapshot for this combo run.
    # This gives us point-in-time scores (DWM, WMQ, intraday, futures, etc.)
    # without changing how the "current" file is used by the dashboard.
    # ------------------------------------------------------------------
    # Use UTC timestamp in a file-name-safe format (no ":" characters)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

    hist_dir = DATA / "combo_history" / namespace / combo_name
    # For local backend, storage.save_parquet will handle parent dirs;
    # but for clarity we still make the local dirs here.
    hist_dir.mkdir(parents=True, exist_ok=True)

    hist_path = hist_dir / f"combo_{combo_name}_asof={ts}.parquet"
    """combo_df.to_parquet(hist_path)"""
    storage.save_parquet(combo_df, hist_path)
    print(f"[OK] Wrote combo history snapshot to {hist_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jobs/run_combo.py <namespace> <combo_name>")
        print("Example: python jobs/run_combo.py stocks stocks_c_dwm_shortlist")
        sys.exit(1)

    ns = sys.argv[1]
    combo = sys.argv[2]
    run(ns, combo)
