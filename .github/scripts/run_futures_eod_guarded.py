# .github/scripts/run_futures_eod_guarded.py

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path


# ---- Config: desired local target time + tolerance ----

TARGET_TIME = time(hour=18, minute=15)  # 6:15 pm America/New_York
TOLERANCE_MIN = 30                      # +/- 30 minutes window


def minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute


def main() -> None:
    # Current local time in New York (DST-aware)
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    now_time = now.time()

    now_min = minutes_since_midnight(now_time)
    target_min = minutes_since_midnight(TARGET_TIME)

    diff = abs(now_min - target_min)
    # If you ever move target near midnight, you could use:
    # diff = min(diff, 1440 - diff)
    # For 18:15, the simple diff is fine.

    if diff > TOLERANCE_MIN:
        print(
            f"[INFO] {now} local (NY) is outside +/-{TOLERANCE_MIN} minutes "
            f"of target {TARGET_TIME}. Skipping futures EOD."
        )
        sys.exit(0)

    print(f"[INFO] Within window at {now} local NY. Running futures EOD pipeline...")

    # Repo root (GitHub Actions workspace root == repo root)
    root = Path(__file__).resolve().parents[2]
    print(f"[INFO] Using repo root: {root}")

    # ---- 1) Refresh futures daily/weekly/monthly ----
    cmd_timeframe = [
        sys.executable,
        str(root / "jobs" / "run_timeframe.py"),
        "futures",
        "daily",
        "--cascade",
    ]
    print(f"[INFO] Running: {' '.join(cmd_timeframe)}")
    subprocess.run(cmd_timeframe, check=True)

    # ---- 2) Rebuild the daily-lower futures combo (D/W/M) ----
    # You specified futures_3_dwm_shortlist as the DWM combo
    cmd_dwm = [
        sys.executable,
        str(root / "jobs" / "run_combo.py"),
        "futures",
        "futures_3_dwm_shortlist",
    ]
    print(f"[INFO] Running: {' '.join(cmd_dwm)}")
    subprocess.run(cmd_dwm, check=True)

    print("[OK] Futures EOD profile completed.")


if __name__ == "__main__":
    main()
