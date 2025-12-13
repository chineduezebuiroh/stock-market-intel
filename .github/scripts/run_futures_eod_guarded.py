from __future__ import annotations

# .github/scripts/run_futures_eod_guarded.py

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

# =======================================================
# ---- Config: desired local target time + tolerance ----
# =======================================================
TARGET_TIME = time(hour=18, minute=15)  # 6:15 pm America/New_York
TOLERANCE_MIN = 30                      # +/- 30 minutes window



def minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute


def run_profile() -> None:
    # repo_root: .../stock-market-intel
    """root = Path(__file__).resolve().parents[2]"""
    root = ROOT  # reuse global ROOT

    cmds = [
        # ---------------------------------------------------------
        # 1) Refresh futures daily/weekly/monthly (no Q/Y)
        # ---------------------------------------------------------
        [
            sys.executable,
            str(root / "jobs" / "run_timeframe.py"),
            "futures",
            "daily",
            "--cascade",
        ],

        # ---------------------------------------------------------
        # 2) Rebuild futures MTF combo
        # ---------------------------------------------------------
        #   - D/W/M
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "futures",
            "futures_3_dwm_shortlist",
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
        if df is None or df.empty or len(df) < min_rows:
            raise RuntimeError(
                f"[FATAL] Combo '{combo_name}' invalid: "
                f"{0 if df is None else len(df)} rows (< {min_rows})"
            )

        print(f"[HEALTH] Combo {combo_name} OK ({len(df)} rows)")
    
    # After run_combo calls:
    assert_combo_nonempty("futures_3_dwm_shortlist", min_rows=5)


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
            f"of target {TARGET_TIME}. Skipping futures EOD."
        )
        sys.exit(0)

    print(f"[INFO] Within window at {now} NY. Running futures EOD profile...")
    run_profile()


if __name__ == "__main__":
    main()
