from __future__ import annotations

# .github/scripts/run_futures_intraday_4h_guarded.py

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.guard import now_ny, in_futures_session, run_registry_guarded
from core.intraday_cadence import is_futures_4h_opportunity
from core.health import run_combo_health, print_results

# from core.signal_alerts import notify_on_signals
from core.notify import notify_combo_signals

# =======================================================
# ---- Config: eligible 4h target hours (NY time) ----
# =======================================================
JOB_NAME = "futures_intraday_4h"


def run_profile() -> None:
    root = ROOT

    cmds = [
        # 1) Refresh the canonical 1h data and its direct 4h/daily cascades.
        # This is data preparation for the 4h profile; it deliberately does
        # not run the futures 1h combo, health check, notification, or registry
        # mark.
        [
            sys.executable,
            str(root / "jobs" / "run_timeframe.py"),
            "futures",
            "intraday_1h",
            "--cascade",
        ],
        # 2) Refresh futures weekly for shortlist only
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "futures", "weekly"],
        # 3) Rebuild 4h/D/W combo
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "futures",
            "futures_2_4hdw_shortlist",
        ],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    results = []
    results += run_combo_health(
        combos=["futures_2_4hdw_shortlist"], universe_csv="shortlist_futures.csv"
    )
    print_results(results)

    notify_combo_signals("futures_2_4hdw_shortlist", only_if_changed=True)


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    now = now_ny()

    if event_name == "workflow_dispatch":
        print("[INFO] Manual dispatch; bypassing registry for futures 4h.")
        run_registry_guarded(
            job_name=JOB_NAME,
            fn=run_profile,
            now=now,
            bypass_registry=True,
        )
        return

    if not in_futures_session(now):
        print(f"[INFO] {now} NY outside 4h futures session. Skipping.")
        sys.exit(0)

    if not is_futures_4h_opportunity(now):
        print(f"[INFO] {now} NY not in a Futures 4h opportunity. Skipping.")
        sys.exit(0)

    print(f"[INFO] {now} NY inside 4h cadence window. Running profile...")
    run_registry_guarded(
        job_name=JOB_NAME,
        fn=run_profile,
        now=now,
        bypass_registry=False,
    )


if __name__ == "__main__":
    main()
