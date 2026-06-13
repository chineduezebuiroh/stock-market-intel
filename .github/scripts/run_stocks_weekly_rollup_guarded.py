from __future__ import annotations
# .github/scripts/run_stocks_weekly_rollup_guarded.py

import subprocess
import os
import sys
import pandas as pd
#from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.health import run_combo_health, print_results
from core.guard import run_registry_guarded

from core.notify import notify_combo_signals

from core.paths import DATA, CFG
from core import storage

# =======================================================
# ----- Config: Set job name constant for auidt log -----
# =======================================================
JOB_NAME = "stocks_weekly"

# =======================================================
# Helper Functions
# =======================================================
def missing_shortlist_symbols_for_tf(timeframe: str) -> set[str]:
    shortlist_path = CFG / "shortlist_stocks.csv"
    if not shortlist_path.exists():
        raise FileNotFoundError(f"Missing shortlist file: {shortlist_path}")

    wanted = set(
        pd.read_csv(shortlist_path)["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    snap_path = DATA / f"snapshot_stocks_{timeframe}.parquet"
    if not storage.exists(snap_path):
        return wanted

    snap = storage.load_parquet(snap_path)
    if snap is None or snap.empty or "symbol" not in snap.columns:
        return wanted

    have = set(
        snap["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return wanted - have


def ensure_shortlist_parent_timeframes(root: Path) -> None:
    """
    Weekly rollup always runs quarterly.

    This only bootstraps weekly/monthly if new shortlist symbols were added
    after the normal daily EOD job last refreshed those snapshots.
    """
    for tf in ["weekly", "monthly"]:
        missing = missing_shortlist_symbols_for_tf(tf)

        if not missing:
            print(f"[BOOTSTRAP] stocks:{tf} has all shortlist symbols.")
            continue

        print(
            f"[BOOTSTRAP] stocks:{tf} missing {len(missing)} shortlist symbols: "
            f"{sorted(missing)}"
        )

        cmd = [
            sys.executable,
            str(root / "jobs" / "run_timeframe.py"),
            "stocks",
            tf,
            "--allowed-universes",
            "shortlist_stocks",
        ]

        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

# =======================================================
# Primary Function
# =======================================================
def run_profile() -> None:
    root = ROOT

    ensure_shortlist_parent_timeframes(root)

    cmds = [
        # 1) Refresh stocks QUARTERLY only (no cascade)
        # ---------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "stocks", "quarterly"],

        # 2) Refresh ETF trends on MONTHLY (middle TF for WMQ)
        # ---------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_etf_trends.py"), "monthly"],

        # 3) Rebuild weekly-lower combos (WMQ)
        # ---------------------------------------------------------
        #   - WMQ Shortlist
        [sys.executable, str(root / "jobs" / "run_combo.py"), "stocks", "stocks_b_wmq_shortlist"],
        #   - WMQ Options-eligible
        [sys.executable, str(root / "jobs" / "run_combo.py"), "stocks", "stocks_b_wmq_all"],
    ]


    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    results = []
    results += run_combo_health(combos=["stocks_b_wmq_shortlist"], universe_csv="shortlist_stocks.csv")
    results += run_combo_health(combos=["stocks_b_wmq_all"], universe_csv=None)
    print_results(results)

    #notify_combo_signals("stocks_b_wmq_shortlist", only_if_changed=False)
    #notify_combo_signals("stocks_b_wmq_all", only_if_changed=False)
    notify_combo_signals("stocks_b_wmq_shortlist", only_if_changed=False, route="stocks_weekly")
    notify_combo_signals("stocks_b_wmq_all", only_if_changed=False, route="stocks_weekly")


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # Manual runs:
    # - bypass registry check so you can force a run anytime
    # - still mark successful execution afterward
    if event_name == "workflow_dispatch":
        print("[INFO] Triggered via workflow_dispatch; bypassing registry guard.")
        run_registry_guarded(
            job_name=JOB_NAME,
            fn=run_profile,
            bypass_registry=True,
        )
        return

    # Scheduled runs:
    # - obey execution registry (active flag + last_execution/check_window_hours)
    run_registry_guarded(
        job_name=JOB_NAME,
        fn=run_profile,
        bypass_registry=False,
    )


if __name__ == "__main__":
    main()
