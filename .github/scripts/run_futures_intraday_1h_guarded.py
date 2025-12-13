from __future__ import annotations

# .github/scripts/run_futures_intraday_1h_guarded.py

import os
import subprocess
import sys
from datetime import datetime
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
TZ = ZoneInfo("America/New_York")
MINUTE_TOLERANCE = 5  # around HH:01


def in_futures_session(now: datetime) -> bool:
    dow = now.weekday()  # Mon=0 .. Sun=6
    t = now.time()

    if dow == 5:  # Saturday
        return False

    if dow == 6:  # Sunday
        return t.hour > 18 or (t.hour == 18 and t.minute >= 1)

    if dow in (0, 1, 2, 3):  # Mon–Thu
        return True

    if dow == 4:  # Friday
        # up to 16:01
        return (t.hour < 16) or (t.hour == 16 and t.minute <= 1)

    return False


def near_hour_plus_one(now: datetime) -> bool:
    # We want ~HH:01; if we run every 10 minutes this will fire near 00/10/20/etc.
    # We'll allow ±5 minutes around minute=1.
    minute = now.minute
    return abs(minute - 1) <= MINUTE_TOLERANCE


def run_profile() -> None:
    """root = Path(__file__).resolve().parents[2]"""
    root = ROOT  # reuse global ROOT

    cmds = [
        # 1) Refresh futures intraday_1h + cascade (4h, daily) for shortlist only
        [
            sys.executable,
            str(root / "jobs" / "run_timeframe.py"),
            "futures",
            "intraday_1h",
            "--cascade",
            #"--allowed-universes",
            #"shortlist_futures",
        ],
        # 2) Rebuild 1h/4h/D combo
        [
            sys.executable,
            str(root / "jobs" / "run_combo.py"),
            "futures",
            "futures_1_1h4hd_shortlist",
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
    assert_combo_nonempty("futures_1_1h4hd_shortlist", min_rows=5)


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # Manual runs always execute
    if event_name == "workflow_dispatch":
        print("[INFO] Manual dispatch; running futures 1h intraday profile.")
        run_profile()
        return

    now = datetime.now(TZ)

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
