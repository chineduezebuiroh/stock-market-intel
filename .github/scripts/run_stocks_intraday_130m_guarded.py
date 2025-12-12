from __future__ import annotations

# .github/scripts/run_stocks_intraday_130m_guarded.py

import os
import subprocess
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path

# 130m targets (local NY time)
TARGET_TIMES = [
    time(9, 31),
    time(11, 41),
    time(13, 51),
    time(16, 1),
]
TOLERANCE_MIN = 5  # +/- 5 minutes
TZ = ZoneInfo("America/New_York")


def minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute


def is_trading_day(dt: datetime) -> bool:
    # Mon=0 .. Sun=6
    return dt.weekday() <= 4


def is_within_any_target(now: datetime) -> bool:
    if not is_trading_day(now):
        return False

    now_t = now.time().replace(second=0, microsecond=0)
    now_min = minutes_since_midnight(now_t)

    for target in TARGET_TIMES:
        diff = abs(now_min - minutes_since_midnight(target))
        if diff <= TOLERANCE_MIN:
            return True

    return False


def run_profile() -> None:
    # repo root (this file is .github/scripts/...)
    root = Path(__file__).resolve().parents[2]

    cmds = [
        # 1) Refresh intraday_130m + cascade to D/W for **shortlist only**
        [
            sys.executable,
            str(root / "jobs" / "run_timeframe.py"),
            "stocks",
            "intraday_130m",
            "--cascade",
            "--allowed-universes",
            "shortlist_stocks",
        ],
        # 2) Rebuild 130m/D/W shortlist combo
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "stocks",
            "stocks_d_130mdw_shortlist",
        ],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    def assert_combo_nonempty(combo_name: str, min_rows: int = 10):
        path = DATA / f"combo_{combo_name}.parquet"
        if not storage.exists(path):
            raise RuntimeError(f"[FATAL] Combo file missing: {path}")
        df = storage.load_parquet(path)
        if len(df) < min_rows:
            raise RuntimeError(
                f"[FATAL] Combo {combo_name} too small: {len(df)} rows (< {min_rows})"
            )
    
    # After run_combo calls:
    assert_combo_nonempty("stocks_d_130mdw_shortlist", min_rows=5)


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # Manual runs always execute
    if event_name == "workflow_dispatch":
        print("[INFO] Manual dispatch; running stocks intraday 130m profile.")
        run_profile()
        return

    now = datetime.now(TZ)
    if not is_within_any_target(now):
        print(f"[INFO] {now} NY is not within +/-{TOLERANCE_MIN} min of 130m targets. Skipping.")
        sys.exit(0)

    print(f"[INFO] {now} NY is within intraday 130m window. Running profile...")
    run_profile()


if __name__ == "__main__":
    main()
