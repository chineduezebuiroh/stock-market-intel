from __future__ import annotations

# .github/scripts/run_futures_intraday_1h_guarded.py

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.guard import NY_TZ, now_ny  # ✅ central TZ + now helper
from core.health import run_combo_health, print_results
#from core.signal_alerts import notify_on_signals
from core.notify import notify_combo_signals

# =======================================================
# ---- Config: desired cadence tolerance ----
# =======================================================
MINUTE_TOLERANCE = 25  # around HH:01


def in_futures_session(now: datetime) -> bool:
    """
    CME futures week:
      - Closed Sat all day
      - Opens Sun 6:00pm ET (you’re using 18:01)
      - Closes Fri ~5:00pm ET (you’re using 16:01)
    """
    dow = now.weekday()  # Mon=0 .. Sun=6
    t = now.time()

    if dow == 5:  # Saturday
        return False

    if dow == 6:  # Sunday
        return t.hour > 18 or (t.hour == 18 and t.minute >= 1)

    if dow in (0, 1, 2, 3):  # Mon–Thu
        return True

    if dow == 4:  # Friday (stop around 16:01)
        return (t.hour < 16) or (t.hour == 16 and t.minute <= 1)

    return False


def near_hour_plus_one(now: datetime) -> bool:
    # We want ~HH:01; allow ±5 minutes around minute=1.
    #return abs(now.minute - 1) <= MINUTE_TOLERANCE
    return now.minute - 1 <= MINUTE_TOLERANCE


def run_profile() -> None:
    root = ROOT

    cmds = [
        # 1) Refresh futures intraday_1h + cascade (4h, daily) for shortlist only
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "futures", "intraday_1h", "--cascade"],
        
        # 2) Rebuild 1h/4h/D combo
        [sys.executable, str(root / "jobs" / "run_combo.py"), "futures", "futures_1_1h4hd_shortlist"],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    results = []
    results += run_combo_health(combos=["futures_1_1h4hd_shortlist"], universe_csv="shortlist_futures.csv")
    print_results(results)

    notify_combo_signals("futures_1_1h4hd_shortlist", only_if_changed=True)


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # Manual runs always execute
    if event_name == "workflow_dispatch":
        print("[INFO] Manual dispatch; running futures 1h intraday profile.")
        run_profile()
        return

    now = now_ny()  # ✅ tz-aware NY now

    if not in_futures_session(now):
        print(f"[INFO] {now} NY outside futures weekly session. Skipping.")
        sys.exit(0)

    if not near_hour_plus_one(now):
        print(f"[INFO] {now} NY not near HH:01 window. Skipping.")
        sys.exit(0)

    print(f"[INFO] {now} NY inside 1h futures session + cadence. Running profile...")
    run_profile()


if __name__ == "__main__":
    main()
