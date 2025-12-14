from __future__ import annotations

# .github/scripts/rrun_stocks_monthly_rollup_guarded.py

import subprocess
import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import DATA
from core import storage
from core.health import run_combo_health, print_results

# =======================================================
# ---- Config: desired local target time + tolerance ----
# =======================================================
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


def run_profile() -> None:
    # repo_root: .../stock-market-intel
    """root = Path(__file__).resolve().parents[2]"""
    root = ROOT  # reuse global ROOT

    cmds = [
        # ---------------------------------------------------------
        # 1) Refresh stocks YEARLY only (no cascade)
        # ---------------------------------------------------------
        [
            sys.executable,
            str(root / "jobs" / "run_timeframe.py"),
            "stocks",
            "yearly",
        ],

        # ---------------------------------------------------------
        # 2) Refresh ETF trends on QUARTERLY (middle TF for MQY)
        # ---------------------------------------------------------
        [
            sys.executable,
            str(root / "jobs" / "run_etf_trends.py"),
            "quarterly",
        ],

        # ---------------------------------------------------------
        # 3) Rebuild monthly-lower combos (MQY)
        # ---------------------------------------------------------
        #   - MQY Shortlist
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "stocks",
            "stocks_a_mqy_shortlist",
        ],
        #   - MQY Options-eligible
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "stocks",
            "stocks_a_mqy_all",
        ],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    """
    def assert_combo_nonempty(combo_name: str, min_rows: int = 10):
        path = DATA / f"combo_{combo_name}.parquet"
        if not storage.exists(path):
            raise RuntimeError(f"[FATAL] Combo file missing: {path}")
        df = storage.load_parquet(path)
        if df is None or df.empty or len(df) < min_rows:
            raise RuntimeError(
                f"[FATAL] Combo '{combo_name}' invalid: "
                f"{0 if df is None else len(df)} rows (< {min_rows})"
            )

        print(f"[HEALTH] Combo {combo_name} OK ({len(df)} rows)")
    
    # After run_combo calls:
    assert_combo_nonempty("stocks_a_mqy_shortlist", min_rows=5)
    assert_combo_nonempty("stocks_a_mqy_all", min_rows=5)
    """

    results = []
    results += run_combo_health(combos=["stocks_a_mqy_shortlist"], universe_csv="shortlist_stocks.csv")
    results += run_combo_health(combos=["stocks_a_mqy_all"], universe_csv=None)
    print_results(results)


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
            f"of target {TARGET_TIME}. Skipping monthly stocks rollup."
        )
        sys.exit(0)

    print(f"[INFO] Within window at {now} NY. Running monthly stocks rollup profile...")
    run_profile()


if __name__ == "__main__":
    main()
