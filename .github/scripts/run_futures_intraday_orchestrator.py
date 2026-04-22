from __future__ import annotations

# scripts/run_futures_intraday_orchestrator.py

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.guard import now_ny, should_run_from_registry

# Import the two guard modules (so we can call their gate funcs + run_profile)
import run_futures_intraday_1h_guarded as g1h
import run_futures_intraday_4h_guarded as g4h


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # Manual dispatch: let each guarded module handle its own bypass logic
    if event_name == "workflow_dispatch":
        print("[INFO] Manual dispatch; delegating to 1h + 4h guarded mains.")
        print(f"[INFO] Running 1h guarded main...")
        g1h.main()
        print(f"[ORCH] Running 4h profile...")
        g4h.main()
        return

    # Scheduled dispatch
    _run_if_ready()


def _run_if_ready() -> None:
    now = now_ny()

    # --------------------------------------------------
    # 1) Standalone 1h branch (registry-controlled)
    # --------------------------------------------------
    if g1h.in_futures_session(now):
        if g1h.near_hour_plus_one(now) and not g4h.near_4h_grid(now):
            ok_1h, reason_1h = should_run_from_registry(job_name="futures_intraday_1h", now=now)
            if ok_1h:
                print(f"[ORCH] {now} 1h qualifies and registry allows run. Running standalone 1h profile...")
                g1h.run_profile()
                mark_registry_execution(job_name="futures_intraday_1h", now=now)
            else:
                print(f"[ORCH] 1h qualifies but registry says skip: {reason_1h}")
        else:
            print(f"[ORCH] {now} does not qualify for 1h cadence.")
    else:
        print(f"[ORCH] {now} outside 1h futures session.")

    # --------------------------------------------------
    # 2) 4h branch (registry-controlled, with 1h dependency refresh)
    # --------------------------------------------------
    if g4h.in_futures_session(now):
        if g4h.near_4h_grid(now):
            ok_4h, reason_4h = should_run_from_registry(job_name="futures_intraday_4h", now=now)
            if ok_4h:
                print(f"[ORCH] {now} 4h qualifies and registry allows run.")
                print("[ORCH] Running 1h dependency refresh before 4h...")
                g1h.run_profile()   # dependency step, NOT standalone 1h job
                print("[ORCH] Running 4h profile...")
                g4h.run_profile()
                mark_registry_execution(job_name="futures_intraday_4h", now=now)
            else:
                print(f"[ORCH] 4h qualifies but registry says skip: {reason_4h}")
        else:
            print(f"[ORCH] {now} does not qualify for 4h cadence.")
    else:
        print(f"[ORCH] {now} outside 4h futures session.")


if __name__ == "__main__":
    main()
