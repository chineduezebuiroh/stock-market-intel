from __future__ import annotations

# .github/scripts/run_stocks_eod_guarded.py

import subprocess
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path


# ---- Config: desired local target time + tolerance ----

TARGET_TIME = time(hour=16, minute=15)  # 4:15 pm America/New_York
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
    # Wrap-around not needed for 16:15, but we keep this simple case
    # If you ever target around midnight, you'd do: diff = min(diff, 1440 - diff)

    if diff > TOLERANCE_MIN:
        print(
            f"[INFO] {now} local (NY) is outside +/-{TOLERANCE_MIN} minutes "
            f"of target {TARGET_TIME}. Skipping stocks EOD."
        )
        sys.exit(0)

    print(f"[INFO] Within window at {now} local NY. Running stocks EOD pipeline...")

    # Repo root (actions workspace is repo root)
    root = Path(__file__).resolve().parents[2]
    print(f"[INFO] Using repo root: {root}")

    # ---- 1) Refresh stocks daily/weekly/monthly (no Q/Y) ----
    cmd_timeframe = [
        sys.executable,
        str(root / "jobs" / "run_timeframe.py"),
        "stocks",
        "daily",
        "--cascade",
    ]
    print(f"[INFO] Running: {' '.join(cmd_timeframe)}")
    subprocess.run(cmd_timeframe, check=True)

    # ---- 2) Rebuild daily-lower combos that depend on fresh D/W/M ----
    #   - DWM Shortlist
    cmd_dwm_short = [
        sys.executable,
        str(root / "jobs" / "run_combo.py"),
        "stocks",
        "stocks_c_dwm_shortlist",
    ]
    print(f"[INFO] Running: {' '.join(cmd_dwm_short)}")
    subprocess.run(cmd_dwm_short, check=True)

    #   - DWM Options-eligible
    cmd_dwm_all = [
        sys.executable,
        str(root / "jobs" / "run_combo.py"),
        "stocks",
        "stocks_c_dwm_all",
    ]
    print(f"[INFO] Running: {' '.join(cmd_dwm_all)}")
    subprocess.run(cmd_dwm_all, check=True)

    print("[OK] Stocks EOD profile completed.")


if __name__ == "__main__":
    main()
