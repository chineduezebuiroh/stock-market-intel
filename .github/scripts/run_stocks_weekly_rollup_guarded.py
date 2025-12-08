from __future__ import annotations

# .github/scripts/run_stocks_weekly_rollup_guarded.py

import subprocess
import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path


# =======================================================
# ---- Config: desired local target time + tolerance ----
# =======================================================
TARGET_TIME = time(hour=6, minute=30)  # 6:30 am America/New_York (Sat early)
TOLERANCE_MIN = 45                    # +/- 45 minutes window



def minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute



def run_profile() -> None:
    # repo_root: .../stock-market-intel
    root = Path(__file__).resolve().parents[2]

    cmds = [
        # ---------------------------------------------------------
        # 1) Refresh stocks QUARTERLY only (no cascade)
        # ---------------------------------------------------------
        [
            sys.executable,
            str(root / "jobs" / "run_timeframe.py"),
            "stocks",
            "quarterly",
        ],

        # ---------------------------------------------------------
        # 2) Refresh ETF trends on MONTHLY (middle TF for WMQ)
        # ---------------------------------------------------------
        [
            sys.executable,
            str(root / "jobs" / "run_etf_trends.py"),
            "monthly",
        ],

        # ---------------------------------------------------------
        # 3) Rebuild weekly-lower combos (WMQ)
        # ---------------------------------------------------------
        #   - WMQ Shortlist
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "stocks",
            "stocks_b_wmq_shortlist",
        ],
        #   - WMQ Options-eligible
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "stocks",
            "stocks_b_wmq_all",
        ],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)



def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # ✅ Manual runs always execute, regardless of time
    if event_name == "workflow_dispatch":
        print("[INFO] Triggered via workflow_dispatch; bypassing time-window guard.")
        run_profile()
        return

    # ✅ Scheduled runs: enforce DST-aware time window
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    now_time = now.time()

    now_min = minutes_since_midnight(now_time)
    target_min = minutes_since_midnight(TARGET_TIME)
    diff = abs(now_min - target_min)

    if diff > TOLERANCE_MIN:
        print(
            f"[INFO] {now} local (NY) is outside +/-{TOLERANCE_MIN} minutes "
            f"of target {TARGET_TIME}. Skipping weekly stocks rollup."
        )
        sys.exit(0)

    print(f"[INFO] Within window at {now} NY. Running weekly stocks rollup profile...")
    run_profile()


if __name__ == "__main__":
    main()
