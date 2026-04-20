from __future__ import annotations

# .github/scripts/run_futures_eod_guarded.py

import subprocess
import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.health import run_combo_health, print_results
from core.guard import run_registry_guarded
#from core.signal_alerts import notify_on_signals
from core.notify import notify_combo_signals

# =======================================================
# ----- Config: Set job name constant for auidt log -----
# =======================================================
JOB_NAME = "futures_eod"


def run_profile() -> None:
    root = ROOT

    cmds = [
        # 1) Refresh futures 1h so it's canonically ready
        # ---------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "futures", "intraday_1h"],
        
        # 2) Refresh futures daily/weekly/monthly (no Q/Y)
        # ---------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "futures", "daily", "--cascade"],

        # 3) Rebuild futures MTF combo
        # ---------------------------------------------------------
        #   - D/W/M
        [sys.executable, str(root / "jobs" / "run_combo.py"), "futures", "futures_3_dwm_shortlist"],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    results = []
    results += run_combo_health(combos=["futures_3_dwm_shortlist"], universe_csv="shortlist_futures.csv")
    print_results(results)

    notify_combo_signals("futures_3_dwm_shortlist", only_if_changed=False)


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # ✅ Manual runs:
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

    # ✅ Scheduled runs:
    # - obey execution registry (active flag + last_execution/check_window_hours)
    run_registry_guarded(
        job_name=JOB_NAME,
        fn=run_profile,
        bypass_registry=False,
    )

if __name__ == "__main__":
    main()
