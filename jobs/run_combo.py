import sys
from pathlib import Path

import pandas as pd
import yaml

from etl.window import parquet_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CFG = ROOT / "config"


def load_multi_tf_config() -> dict:
    """
    Load the multi-timeframe combos config.
    """
    path = CFG / "multi_timeframe_combos.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def symbols_for(universe: str) -> list[str]:
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


def load_role_frame(
    namespace: str,
    timeframe: str,
    symbols: list[str],
    role_prefix: str,
) -> pd.DataFrame:
    """
    Load the latest bar for each symbol for a given namespace+timeframe,
    and prefix all columns with role_prefix (e.g., 'lower_', 'upper_').
    Index will be symbol.
    """
    rows = []
    for sym in symbols:
        p = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
        if not p.exists():
            continue

        df = pd.read_parquet(p)
        if df.empty:
            continue

        last = df.iloc[-1:].copy()
        last["symbol"] = sym
        rows.append(last)

    if not rows:
        return pd.DataFrame()

    frame = pd.concat(rows, axis=0)
    frame = frame.set_index("symbol")

    # Prefix all columns with the role to avoid collisions
    frame = frame.rename(columns={c: f"{role_prefix}{c}" for c in frame.columns})
    return frame


def build_combo_df(namespace: str, combo_name: str, cfg: dict) -> pd.DataFrame:
    """
    For a given combo, load lower/middle/upper latest bars and join them.
    """
    ns_cfg = cfg.get(namespace, {})
    combo_cfg = ns_cfg.get(combo_name)
    if combo_cfg is None:
        raise ValueError(f"Combo '{combo_name}' not found under namespace '{namespace}'")

    lower_tf = combo_cfg["lower_tf"]
    middle_tf = combo_cfg.get("middle_tf")
    upper_tf = combo_cfg["upper_tf"]
    universe = combo_cfg["universe"]

    symbols = symbols_for(universe)
    if not symbols:
        print(f"[WARN] No symbols for universe '{universe}'.")
        return pd.DataFrame()

    lower_df = load_role_frame(namespace, lower_tf, symbols, "lower_")
    if lower_df.empty:
        print(f"[WARN] No lower timeframe data for combo '{combo_name}'.")
        return pd.DataFrame()

    frames = [lower_df]

    if middle_tf and middle_tf.lower() != "null":
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

    # Inner join on symbol index
    combo_df = frames[0]
    for f in frames[1:]:
        combo_df = combo_df.join(f, how="inner")

    if combo_df.empty:
        print(f"[WARN] Joined combo dataframe for '{combo_name}' is empty.")
        return combo_df

    return combo_df


def basic_signal_logic(combo_df: pd.DataFrame) -> pd.DataFrame:
    """
    First-pass placeholder MTF logic.
    We'll refine this later to match your exact TOS rules.

    For now:
      - LONG if upper_ema_8 > upper_ema_21,
               lower_ema_8 > lower_ema_21,
               40 < lower_rsi_14 < 70
      - SHORT if upper_ema_8 < upper_ema_21,
                lower_ema_8 < lower_ema_21,
                30 < lower_rsi_14 < 60
      - otherwise WATCH.
    """
    df = combo_df.copy()

    def has_cols(cols: list[str]) -> bool:
        return all(c in df.columns for c in cols)

    # Defaults: everything is "watch" until promoted
    df["signal"] = "watch"

    # Long logic
    if has_cols(["upper_ema_8", "upper_ema_21", "lower_ema_8", "lower_ema_21", "lower_rsi_14"]):
        long_mask = (
            (df["upper_ema_8"] > df["upper_ema_21"])
            & (df["lower_ema_8"] > df["lower_ema_21"])
            & (df["lower_rsi_14"] > 40)
            & (df["lower_rsi_14"] < 70)
        )
        df.loc[long_mask, "signal"] = "long"

    # Short logic
    if has_cols(["upper_ema_8", "upper_ema_21", "lower_ema_8", "lower_ema_21", "lower_rsi_14"]):
        short_mask = (
            (df["upper_ema_8"] < df["upper_ema_21"])
            & (df["lower_ema_8"] < df["lower_ema_21"])
            & (df["lower_rsi_14"] > 30)
            & (df["lower_rsi_14"] < 60)
        )
        # Only overwrite "watch" entries
        df.loc[short_mask & (df["signal"] == "watch"), "signal"] = "short"

    return df


def run_combo(namespace: str, combo_name: str):
    """
    Entry point: build the combo dataframe, apply signal logic, and persist snapshot.
    """
    cfg = load_multi_tf_config()
    combo_df = build_combo_df(namespace, combo_name, cfg)
    if combo_df.empty:
        print(f"[INFO] No data for combo '{namespace}:{combo_name}'. Nothing to write.")
        return

    combo_df = basic_signal_logic(combo_df)

    # Move index 'symbol' to a column for easier use in Streamlit
    out = combo_df.copy()
    out = out.reset_index().rename(columns={"index": "symbol"})
    out["combo"] = combo_name

    out_path = DATA / f"snapshot_combo_{namespace}_{combo_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"[OK] Wrote combo snapshot: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jobs/run_combo.py <namespace> <combo_name>")
        print("Example: python jobs/run_combo.py stocks stocks_c_dm_shortlist")
        sys.exit(1)

    ns = sys.argv[1]
    combo = sys.argv[2]
    run_combo(ns, combo)
