from __future__ import annotations

# .github/scripts/run_stocks_eod_guarded.py

import subprocess
import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path

from core.paths import DATA
from core import storage
import pandas as pd

# =======================================================
# ---- Config: desired local target time + tolerance ----
# =======================================================
TARGET_TIME = time(hour=16, minute=15)  # 4:15 pm America/New_York
TOLERANCE_MIN = 30                      # +/- 30 minutes window


def minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute


def run_profile() -> None:
    # repo_root: .../stock-market-intel
    # this script lives in .github/scripts → parents[0]=scripts, [1]=.github, [2]=repo root
    root = Path(__file__).resolve().parents[2]

    cmds = [
        # ---------------------------------------------------------
        # 1) Refresh stocks daily/weekly/monthly (no Q/Y)
        # ---------------------------------------------------------
        [
            sys.executable,
            str(root / "jobs" / "run_timeframe.py"),
            "stocks",
            "daily",
            "--cascade",
        ],

        # ----------------------------------------------------------------------
        # 2) Refresh ETF trends on weekly (middle) and daily (lower) timeframes
        # ----------------------------------------------------------------------
        [
            sys.executable,
            str(root / "jobs" / "run_etf_trends.py"),
            "weekly",
        ],
        
        [
            sys.executable,
            str(root / "jobs" / "run_etf_trends.py"),
            "daily",
        ],

        # ---------------------------------------------------------
        # 3) Rebuild daily-lower combos that depend on fresh D/W/M
        # ---------------------------------------------------------
        #   - DWM Shortlist
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "stocks",
            "stocks_c_dwm_shortlist",
        ],

        #   - DWM Options-eligible
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "stocks",
            "stocks_c_dwm_all",
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
    assert_combo_nonempty("stocks_c_dwm_shortlist", min_rows=5)
    assert_combo_nonempty("stocks_c_dwm_all", min_rows=5)


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
            f"of target {TARGET_TIME}. Skipping stocks EOD."
        )
        sys.exit(0)

    print(f"[INFO] Within window at {now} NY. Running stocks EOD profile...")
    run_profile()


if __name__ == "__main__":
    main()
