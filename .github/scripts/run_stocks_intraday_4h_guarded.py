from __future__ import annotations
# .github/scripts/run_stocks_intraday_4h_guarded.py

import os
import subprocess
import sys
import pandas as pd
from datetime import datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import DATA
from core import storage
from core.guard import run_registry_guarded, now_ny  # ✅ central TZ-aware NY now
from core.health import run_combo_health, print_results
#from core.signal_alerts import notify_on_signals
from core.notify import notify_combo_signals, send_telegram_message

# =======================================================
# ---- Config: 4h targets (NY time) + tolerance ----
# =======================================================
JOB_NAME = "stocks_intraday_4h"
TARGET_TIMES = [
    time(9, 1),
    time(13, 1),
    time(17, 1),
]
TOLERANCE_MIN = 70  # + 70 minutes


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
        #diff = abs(now_min - minutes_since_midnight(target))
        diff = now_min - minutes_since_midnight(target)
        if 0 <= diff <= TOLERANCE_MIN:
            return True

    return False


def run_profile() -> None:
    root = ROOT

    cmds = [
        # 1) Refresh intraday_4h + cascade to D/W for **shortlist only**
        [sys.executable, str(root / "jobs" / "run_timeframe.py"), "stocks", "intraday_4h", "--cascade", "--allowed-universes", "shortlist_stocks"],
        
        # 2) Rebuild 4h/D/W shortlist combo
        [sys.executable, str(root / "jobs" / "run_combo.py"), "stocks", "stocks_d_4hdw_shortlist"],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # =======================================================
    #  Telemetry Reporting Handling
    # =======================================================
    def notify_repair_telemetry(warn_count: int = 50, warn_pct: float = 0.15) -> None:
        p = DATA / "_health" / "stocks_intraday_4h_repair_telemetry.parquet"
    
        if not storage.exists(p):
            print("[HEALTH] no repair telemetry artifact found.")
            return
    
        df = storage.load_parquet(p)
        if df is None or df.empty:
            print("[HEALTH] repair telemetry empty.")
            return
    
        df["bars_total"] = pd.to_numeric(df["bars_total"], errors="coerce").fillna(0)
        df["bars_repaired"] = pd.to_numeric(df["bars_repaired"], errors="coerce").fillna(0)
        df["repair_pct"] = df["bars_repaired"] / df["bars_total"].replace(0, pd.NA)
    
        bad = df[
            (df["bars_repaired"] >= warn_count)
            | (df["repair_pct"] >= warn_pct)
        ].copy()
    
        if bad.empty:
            print("[HEALTH] repair telemetry below warning thresholds.")
            return
    
        bad = bad.sort_values("bars_repaired", ascending=False).head(20)
    
        lines = ["⚠️ *Stocks Intraday 4h Repair Telemetry*"]
        for _, r in bad.iterrows():
            lines.append(
                f"- `{r['symbol']}` {r['source']}: "
                f"{int(r['bars_repaired'])}/{int(r['bars_total'])} repaired "
                f"({float(r['repair_pct']):.1%})"
            )
    
        send_telegram_message("\n".join(lines))

    notify_repair_telemetry()

    # =======================================================
    #  HEALTH CHECK SECTION — FAIL LOUDLY IF COMBOS ARE BAD
    # =======================================================
    results = []
    # ✅ fix: health-check the same combo name you just built ("4hdw", not "4dw")
    results += run_combo_health(combos=["stocks_d_4hdw_shortlist"], universe_csv="shortlist_stocks.csv")
    print_results(results)

    notify_combo_signals("stocks_d_4hdw_shortlist", only_if_changed=True)


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    now = now_ny()

    # Manual runs always execute
    if event_name == "workflow_dispatch":
        print("[INFO] Manual dispatch; bypassing registry for stocks intraday 4h.")
        run_registry_guarded(
            job_name=JOB_NAME,
            fn=run_profile,
            now=now,
            bypass_registry=True,
        )
        return
    
    if not is_within_any_target(now):
        print(
            f"[INFO] {now} NY is not within +/-{TOLERANCE_MIN} min of 4h targets. Skipping."
        )
        sys.exit(0)

    print(f"[INFO] {now} NY is within intraday 4h window. Running profile...")
    run_registry_guarded(
        job_name=JOB_NAME,
        fn=run_profile,
        now=now,
        bypass_registry=False,
    )

if __name__ == "__main__":
    main()
