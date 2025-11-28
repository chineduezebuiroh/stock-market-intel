import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from etl.window import parquet_path

DATA = ROOT / "data"
CFG = ROOT / "config"

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
    For a given timeframe + list of symbols, load the *latest bar* for each symbol
    and return a DataFrame indexed by symbol with columns like:

        lower_open, lower_high, ..., lower_atr_14

    One row per symbol, flat column names.
    """
    rows = []

    for sym in symbols:
        p = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        if not p.exists():
            continue

        df = pd.read_parquet(p)
        if df.empty:
            continue

        last = df.iloc[-1]

        row = {"symbol": sym}
        for col in BASE_COLS:
            if col in last.index:
                row[f"{role_prefix}{col}"] = last[col]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("symbol")
    out.columns = out.columns.astype(str)
    return out


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


def basic_signal_logic(combo_df: pd.DataFrame) -> pd.DataFrame:
    """
    First-pass placeholder MTF logic.
    Right now, just tags everything as 'watch'.
    We'll replace this with your real EMA/SMA/Wilder logic later.
    """
    df = combo_df.copy()
    df["signal"] = "watch"
    return df


def run_combo(namespace: str, combo_name: str):
    """Entry point: build combo dataframe, apply signal logic, persist snapshot."""
    cfg = load_multi_tf_config()
    df = build_combo_df(namespace, combo_name, cfg)
    if df.empty:
        print(f"[INFO] No data for combo '{namespace}:{combo_name}'. Nothing to write.")
        return

    out = basic_signal_logic(df)
    out = out.reset_index()  # bring 'symbol' out of index
    out["combo"] = combo_name

    out_path = DATA / f"combo_{namespace}_{combo_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"[OK] Wrote combo snapshot to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jobs/run_combo.py <namespace> <combo_name>")
        print("Example: python jobs/run_combo.py stocks stocks_c_dwm_shortlist")
        sys.exit(1)

    ns = sys.argv[1]
    combo = sys.argv[2]
    run_combo(ns, combo)
