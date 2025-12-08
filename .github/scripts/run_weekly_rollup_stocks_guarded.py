from __future__ import annotations

# .github/scripts/run_weekly_rollup_stocks_guarded.py

import subprocess
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path

TARGET_TIME = time(hour=6, minute=30)  # 6:30 am America/New_York (Sat early)
TOLERANCE_MIN = 45                    # +/- 45 minutes window


def minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute


def main() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    now_time = now.time()
    today = now.date()

    # Python weekday: Monday=0 ... Sunday=6
    weekday = now.weekday()

    now_min = minutes_since_midnight(now_time)
    target_min = minutes_since_midnight(TARGET_TIME)
    diff = abs(now_min - target_min)

    if weekday != 5:  # 5 = Saturday
        print(f"[INFO] Today ({today}) is not Saturday in NY. Skipping weekly rollup.")
        sys.exit(0)

    if diff > TOLERANCE_MIN:
        print(
            f"[INFO] {now} local (NY) is outside +/-{TOLERANCE_MIN} minutes "
            f"of target {TARGET_TIME}. Skipping weekly rollup."
        )
        sys.exit(0)

    print(f"[INFO] Within weekly window at {now} local NY. Running weekly rollup pipeline...")

    root = Path(__file__).resolve().parents[2]
    print(f"[INFO] Using repo root: {root}")

    # ------------------------------------------------
    # ---- 1) Refresh quarterly bars (no cascade) ----
    # ------------------------------------------------
    cmd_quarterly = [
        sys.executable,
        str(root / "jobs" / "run_timeframe.py"),
        "stocks",
        "quarterly",
    ]
    print(f"[INFO] Running: {' '.join(cmd_quarterly)}")
    subprocess.run(cmd_quarterly, check=True)

    # ---------------------------------------------------------
    # ---- 2) Refresh ETF trends on weekly (middle) timreframe
    # ---------------------------------------------------------
    cmd_etf_timeframe = [
        sys.executable,
        str(root / "jobs" / "run_etf_trends.py"),
        "monthly",
    ]
    print(f"[INFO] Running: {' '.join(cmd_etf_timeframe)}")
    subprocess.run(cmd_etf_timeframe, check=True)

    # ------------------------------------------------
    # ---- 3) Rebuild WMQ combos ----
    # ------------------------------------------------
    # Shortlist WMQ
    cmd_wmq_short = [
        sys.executable,
        str(root / "jobs" / "run_combo.py"),
        "stocks",
        "stocks_b_wmq_shortlist",
    ]
    print(f"[INFO] Running: {' '.join(cmd_wmq_short)}")
    subprocess.run(cmd_wmq_short, check=True)

    # Options-eligible WMQ
    cmd_wmq_all = [
        sys.executable,
        str(root / "jobs" / "run_combo.py"),
        "stocks",
        "stocks_b_wmq_all",
    ]
    print(f"[INFO] Running: {' '.join(cmd_wmq_all)}")
    subprocess.run(cmd_wmq_all, check=True)

    print("[OK] Weekly rollup (WMQ) completed.")


if __name__ == "__main__":
    main()
