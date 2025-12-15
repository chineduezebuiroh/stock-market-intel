from __future__ import annotations

# .github/scripts/run_stocks_monthly_rollup_guarded.py

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
from core.guard import run_guarded  # ✅ central guard logic
from core.signal_alerts import notify_on_signals

# =======================================================
# ---- Config: desired local target time + tolerance ----
# =======================================================
TARGET_TIME = time(hour=6, minute=30)  # 6:30 am America/New_York (Sunday early)
TOLERANCE_MIN = 45                    # +/- 45 minutes window


def is_first_sunday(dt: datetime) -> bool:
    """
    Return True if the given datetime (in local TZ) is the first Sunday of the month.
    First Sunday = any Sunday with day-of-month <= 7.
    """
    if dt.weekday() != 6:  # Sunday = 6
        return False
    return dt.day <= 7



def run_profile() -> None:
    root = ROOT

    cmds = [
        # 1) Refresh stocks YEARLY only (no cascade)
        # ---------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "stocks", "yearly"],

        # 2) Refresh ETF trends on QUARTERLY (middle TF for MQY)
        # ---------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_etf_trends.py"), "quarterly"],

        # 3) Rebuild monthly-lower combos (MQY)
        # ---------------------------------------------------------
        #   - MQY Shortlist
        [sys.executable, str(root / "jobs" / "run_combo.py"), "stocks", "stocks_a_mqy_shortlist"],
        #   - MQY Options-eligible
        [sys.executable, str(root / "jobs" / "run_combo.py"), "stocks", "stocks_a_mqy_all"],
    ]
    
    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    results = []
    results += run_combo_health(combos=["stocks_a_mqy_shortlist"], universe_csv="shortlist_stocks.csv")
    results += run_combo_health(combos=["stocks_a_mqy_all"], universe_csv=None)
    print_results(results)

    notify_on_signals(
        combos=["stocks_a_mqy_shortlist", "stocks_a_mqy_all"],
        title="Futures Monthly Signals",
        changed_only=False,
        changed_group="stocks_monthly",
    )


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # We want "NY time" for the first-Sunday gate (explicit + obvious here)
    now_ny = datetime.now(ZoneInfo("America/New_York"))

    # ✅ Manual runs bypass first-Sunday + time window (still idempotent by default)
    if event_name == "workflow_dispatch":
        print("[INFO] Triggered via workflow_dispatch; bypassing first-Sunday + time-window guard.")
        run_guarded(
            marker_name="stocks_monthly_rollup",
            period="monthly",
            target_time=TARGET_TIME,
            tolerance_min=TOLERANCE_MIN,
            mode="abs",                 # monthly/weekly weekend jobs: abs window is fine
            fn=run_profile,
            bypass_time_window=True,
            respect_idempotency=False,
        )
        return

    # ✅ Scheduled runs: first-Sunday gate (local), then time/idempotency (core)
    if not is_first_sunday(now_ny):
        print(
            f"[INFO] Today ({now_ny.date()}) is not the first Sunday of the month in NY. "
            "Skipping monthly rollup."
        )
        sys.exit(0)

    run_guarded(
        marker_name="stocks_monthly_rollup",
        period="monthly",
        target_time=TARGET_TIME,
        tolerance_min=TOLERANCE_MIN,
        mode="abs",
        fn=run_profile,
        bypass_time_window=False,
        respect_idempotency=True,
    )


if __name__ == "__main__":
    main()
