from __future__ import annotations

# .github/scripts/run_monthly_rollup_stocks_guarded.py

import subprocess
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path

TARGET_TIME = time(hour=6, minute=30)  # 6:30 am America/New_York (Sunday early)
TOLERANCE_MIN = 45                    # +/- 45 minutes window


def minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute


def is_first_sunday(dt: datetime) -> bool:
    """
    Return True if the given datetime (in local TZ) is the first Sunday of the month.
    First Sunday = any Sunday with day-of-month <= 7.
    """
    if dt.weekday() != 6:  # Sunday = 6
        return False
    return dt.day <= 7


def main() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    now_time = now.time()
    today = now.date()

    if not is_first_sunday(now):
        print(f"[INFO] Today ({today}) is not the first Sunday of the month in NY. Skipping monthly rollup.")
        sys.exit(0)

    now_min = minutes_since_midnight(now_time)
    target_min = minutes_since_midnight(TARGET_TIME)
    diff = abs(now_min - target_min)

    if diff > TOLERANCE_MIN:
        print(
            f"[INFO] {now} local (NY) is outside +/-{TOLERANCE_MIN} minutes "
            f"of target {TARGET_TIME}. Skipping monthly rollup."
        )
        sys.exit(0)

    print(f"[INFO] Within monthly window at {now} local NY on first Sunday. Running monthly rollup pipeline...")

    root = Path(__file__).resolve().parents[2]
    print(f"[INFO] Using repo root: {root}")

    # ---------------------------------------------
    # ---- 1) Refresh yearly bars (no cascade) ----
    # ---------------------------------------------
    cmd_yearly = [
        sys.executable,
        str(root / "jobs" / "run_timeframe.py"),
        "stocks",
        "yearly",
    ]
    print(f"[INFO] Running: {' '.join(cmd_yearly)}")
    subprocess.run(cmd_yearly, check=True)

    # ------------------------------------------------------------
    # ---- 2) Refresh ETF trends on quarterly (middle) timreframe
    # ------------------------------------------------------------
    cmd_etf_timeframe = [
        sys.executable,
        str(root / "jobs" / "run_etf_trends.py"),
        "quarterly",
    ]
    print(f"[INFO] Running: {' '.join(cmd_etf_timeframe)}")
    subprocess.run(cmd_etf_timeframe, check=True)

    # ---------------------------------------------
    # ---- 3) Rebuild MQY combos ----
    # ---------------------------------------------
    # Shortlist MQY
    cmd_mqy_short = [
        sys.executable,
        str(root / "jobs" / "run_combo.py"),
        "stocks",
        "stocks_a_mqy_shortlist",
    ]
    print(f"[INFO] Running: {' '.join(cmd_mqy_short)}")
    subprocess.run(cmd_mqy_short, check=True)

    # Options-eligible MQY
    cmd_mqy_all = [
        sys.executable,
        str(root / "jobs" / "run_combo.py"),
        "stocks",
        "stocks_a_mqy_all",
    ]
    print(f"[INFO] Running: {' '.join(cmd_mqy_all)}")
    subprocess.run(cmd_mqy_all, check=True)

    print("[OK] Monthly rollup (MQY) completed.")


if __name__ == "__main__":
    main()
