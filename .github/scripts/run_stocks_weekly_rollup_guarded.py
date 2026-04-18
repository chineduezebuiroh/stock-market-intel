from __future__ import annotations

# .github/scripts/run_stocks_weekly_rollup_guarded.py

import subprocess
import os
import sys
#from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.health import run_combo_health, print_results
from core.guard import run_registry_guarded

from core.notify import notify_combo_signals

# =======================================================
# ----- Config: Set job name constant for auidt log -----
# =======================================================
JOB_NAME = "stocks_weekly"


def run_profile() -> None:
    root = ROOT


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

    notify_combo_signals("stocks_b_wmq_shortlist", only_if_changed=False)
    notify_combo_signals("stocks_b_wmq_all", only_if_changed=False)


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
