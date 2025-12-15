from __future__ import annotations

# .github/scripts/run_futures_intraday_4h_guarded.py

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.guard import now_ny  # ✅ central TZ-aware NY now
from core.health import run_combo_health, print_results

# =======================================================
# ---- Config: cadence window ----
# =======================================================
MINUTE_TOLERANCE = 60
FOUR_HOUR_HOURS = {1, 5, 9, 13, 17, 21}


def in_futures_4h_session(now: datetime) -> bool:
    dow = now.weekday()  # Mon=0..Sun=6
    t = now.time()

    if dow == 5:  # Saturday
        return False

    if dow == 6:  # Sunday: from 17:01 onwards
        return (t.hour > 17) or (t.hour == 17 and t.minute >= 1)

    if dow in (0, 1, 2, 3):  # Mon–Thu: all day
        return True

    if dow == 4:  # Friday: up to 13:01
        return (t.hour < 13) or (t.hour == 13 and t.minute <= 1)

    return False


def near_4h_grid(now: datetime) -> bool:
    t = now.time()

    # allow around HH:01
    #if abs(t.minute - 1) > MINUTE_TOLERANCE:
    if t.minute - 1 > MINUTE_TOLERANCE:
        return False

    dow = now.weekday()
    h = t.hour

    if dow == 6:  # Sunday: only 17:01 and 21:01
        return h in {17, 21}

    if dow in (0, 1, 2, 3):  # Mon–Thu: all
        return h in FOUR_HOUR_HOURS

    if dow == 4:  # Friday: up to 13:01 so only 1,5,9,13
        return h in {1, 5, 9, 13}

    return False














def run_profile() -> None:
    root = ROOT

    cmds = [
        # 1) Refresh futures weekly for shortlist only
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "futures", "weekly"],
        
        # 2) Rebuild 4h/D/W combo
        [sys.executable, str(root / "jobs" / "run_combo.py"), "futures", "futures_2_4hdw_shortlist"],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    results = []
    results += run_combo_health(combos=["futures_2_4hdw_shortlist"], universe_csv="shortlist_futures.csv")
    print_results(results)


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # Manual override
    if event_name == "workflow_dispatch":
        print("[INFO] Manual dispatch; running futures 4h combo profile.")
        run_profile()
        return

    now = now_ny()

    if not in_futures_4h_session(now):
        print(f"[INFO] {now} NY outside 4h futures session. Skipping.")
        sys.exit(0)

    if not near_4h_grid(now):
        print(f"[INFO] {now} NY not on 4h grid. Skipping.")
        sys.exit(0)

    print(f"[INFO] {now} NY inside 4h cadence window. Running profile...")
    run_profile()


if __name__ == "__main__":
    main()
