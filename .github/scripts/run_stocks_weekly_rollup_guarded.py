from __future__ import annotations

# .github/scripts/run_stocks_weekly_rollup_guarded.py

import subprocess
import os
import sys
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.health import run_combo_health, print_results
from core.guard import run_guarded  # ✅ your existing core/guard.py

# =======================================================
# ---- Config: desired local target time + tolerance ----
# =======================================================
TARGET_TIME = time(hour=6, minute=30)   # 6:30am America/New_York
TOLERANCE_MIN = 45                      # +/- 45 minutes


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


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # Manual runs: bypass time window (but you can choose whether to respect idempotency)
    if event_name == "workflow_dispatch":
        print("[INFO] Triggered via workflow_dispatch; bypassing time-window guard.")
        run_guarded(
            marker_name="stocks_weekly_rollup",
            period="weekly",
            target_time=TARGET_TIME,
            tolerance_min=TOLERANCE_MIN,
            mode="abs",                 # ✅ weekly: abs window is fine
            fn=run_profile,
            bypass_time_window=True,
            respect_idempotency=False, 
        )
        return

    # Scheduled runs: enforce window + idempotency
    run_guarded(
        marker_name="stocks_weekly_rollup",
        period="weekly",
        target_time=TARGET_TIME,
        tolerance_min=TOLERANCE_MIN,
        mode="abs",
        fn=run_profile,
        bypass_time_window=False,
        respect_idempotency=True,
    )


if __name__ == "__main__":
    main()
