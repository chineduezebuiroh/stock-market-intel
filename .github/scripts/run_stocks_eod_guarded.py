from __future__ import annotations

# .github/scripts/run_stocks_eod_guarded.py

import subprocess
import os
import sys
#from datetime import time
#from zoneinfo import ZoneInfo
from pathlib import Path
#import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#from core.paths import DATA
#from core import storage
from core.health import run_combo_health, print_results
from core.guard import run_registry_guarded
#from core.signal_alerts import notify_on_signals
from core.notify import notify_combo_signals

# =======================================================
# ----- Config: Set job name constant for auidt log -----
# =======================================================
JOB_NAME = "stocks_eod"


def run_profile() -> None:
    # repo_root: .../stock-market-intel
    """root = Path(__file__).resolve().parents[2]"""
    root = ROOT  # reuse global ROOT

    cmds = [
        # 1) Refresh stocks daily/weekly/monthly (no Q/Y)
        # ---------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "stocks", "daily", "--cascade"],

        # 2) Refresh ETF trends on weekly (middle) and daily (lower) timeframes
        # ----------------------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_etf_trends.py"), "weekly"],
        [sys.executable, str(root / "jobs" / "run_etf_trends.py"), "daily"],

        # 3) Rebuild daily-lower combos that depend on fresh D/W/M
        # ---------------------------------------------------------
        #   - DWM Shortlist
        [sys.executable, str(root / "jobs" / "run_combo.py"), "stocks", "stocks_c_dwm_shortlist"],

        #   - DWM Options-eligible
        [sys.executable, str(root / "jobs" / "run_combo.py"), "stocks", "stocks_c_dwm_all"],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print(f"[INFO] Finished: {' '.join(cmd)} with return code {result.returncode}")
    
    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    results = []
    results += run_combo_health(combos=["stocks_c_dwm_shortlist"], universe_csv="shortlist_stocks.csv")
    results += run_combo_health(combos=["stocks_c_dwm_all"], universe_csv=None)
    print_results(results)

    notify_combo_signals("stocks_c_dwm_shortlist", only_if_changed=False)
    notify_combo_signals("stocks_c_dwm_all", only_if_changed=False)

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
