from __future__ import annotations

# scripts/run_futures_intraday_orchestrator.py

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.guard import now_ny

# Import the two guard modules (so we can call their gate funcs + run_profile)
import run_futures_intraday_1h_guarded as g1h
import run_futures_intraday_4h_guarded as g4h


def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # Manual dispatch: run both checks, run what qualifies
    if event_name == "workflow_dispatch":
        print("[INFO] Manual dispatch; evaluating 1h + 4h gates.")
        _run_if_ready()
        return

    # Scheduled dispatch
    _run_if_ready()


def _run_if_ready() -> None:
    now = now_ny()

    # Shared futures session gate (use the 1h module’s function as the canonical one)
    if not g1h.in_futures_session(now):
        print(f"[INFO] {now} NY outside futures session. Skipping.")
        return

    ran_1h = False

    # 1) Run 1h if it qualifies
    if g1h.near_hour_plus_one(now):
        print(f"[ORCH] {now} NY qualifies for 1h cadence. Running 1h profile...")
        g1h.run_profile()
        ran_1h = True
    else:
        print(f"[ORCH] {now} NY does not qualify for 1h cadence.")

    # 2) Run 4h if it qualifies
    # IMPORTANT: even if it qualifies, we still want it to run AFTER 1h when both qualify
    if g4h.near_4h_grid(now):
        print(f"[ORCH] {now} NY qualifies for 4h cadence.")
        if not ran_1h:
            print("[ORCH] Skipping 4h because 1h did not run this tick.")
            return
            """
            If you want to be more lenient,  allow the job to continue even if ran_1h is False.
            Replace this block:
                - print("[ORCH] Skipping 4h because 1h did not run this tick.")
                - return
            with:
                - print("[ORCH] 4h qualifies but 1h did not run this tick. "
                  "Proceeding anyway (your 4h step assumes 1h cascade has already produced 4h/daily).")
                - # No 'return' statement
            """
        print("[ORCH] Running 4h profile...")
        g4h.run_profile()
    else:
        print(f"[ORCH] {now} NY does not qualify for 4h cadence.")


if __name__ == "__main__":
    main()
