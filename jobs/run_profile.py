from __future__ import annotations

# jobs/run_profile.py

import sys
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------
# Path setup so we can import project modules
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indicators.core import initialize_indicator_engine
from jobs import run_timeframe as rt
from jobs import run_combo as rc

CFG = ROOT / "config"


# -------------------------------------------------------------------
# Profile definitions
# -------------------------------------------------------------------

def profile_daily_eod_stocks() -> None:
    """
    Daily EOD (Mon–Fri) for STOCKS:

    - Refresh stocks daily/weekly/monthly (no quarterly/yearly).
    - Rebuild DAILY-lower combos (stocks DWM).
    """
    print("=== [PROFILE] daily_eod_stocks ===")

    # 1) Timeframes
    print("[STEP] Timeframes: stocks daily --cascade (D/W/M)")
    rt.run(namespace="stocks", timeframe="daily", cascade=True)

    # 2) Combos – stocks DWM (shortlist + options-eligible)
    print("[STEP] Combos: stocks_c_dwm_shortlist")
    rc.run(namespace="stocks", combo_name="stocks_c_dwm_shortlist")

    print("[STEP] Combos: stocks_c_dwm_all")
    rc.run(namespace="stocks", combo_name="stocks_c_dwm_all")

    print("=== [DONE] daily_eod_stocks ===")


def profile_daily_eod_futures() -> None:
    """
    Daily EOD (Sun–Thu) for FUTURES:

    - Refresh futures daily/weekly/monthly.
    - Rebuild DAILY-lower futures combo (futures DWM).
    """
    print("=== [PROFILE] daily_eod_futures ===")

    # 1) Timeframes
    print("[STEP] Timeframes: futures daily --cascade (D/W/M)")
    rt.run(namespace="futures", timeframe="daily", cascade=True)

    # 2) Combos – futures DWM (daily / weekly / monthly)
    print("[STEP] Combos: futures_3_dwm_shortlist")
    rc.run(namespace="futures", combo_name="futures_3_dwm_shortlist")

    print("=== [DONE] daily_eod_futures ===")


def profile_weekly_rollup() -> None:
    """
    Weekly rollup (Fri after close / Sat early) for STOCKS:

    - Refresh stocks quarterly bars.
    - Rebuild WEEKLY-lower combos (stocks WMQ).
    """
    print("=== [PROFILE] weekly_rollup ===")

    # 1) Timeframes
    print("[STEP] Timeframes: stocks quarterly (no cascade)")
    rt.run(namespace="stocks", timeframe="quarterly", cascade=False)

    # 2) Combos – stocks WMQ
    print("[STEP] Combos: stocks_b_wmq_shortlist")
    rc.run(namespace="stocks", combo_name="stocks_b_wmq_shortlist")

    print("[STEP] Combos: stocks_b_wmq_all")
    rc.run(namespace="stocks", combo_name="stocks_b_wmq_all")

    print("=== [DONE] weekly_rollup ===")


def profile_monthly_rollup() -> None:
    """
    Monthly rollup (first Sat night / Sun morning) for STOCKS:

    - Refresh stocks yearly bars.
    - Rebuild MONTHLY-lower combos (stocks MQY).
    """
    print("=== [PROFILE] monthly_rollup ===")

    # 1) Timeframes
    print("[STEP] Timeframes: stocks yearly (no cascade)")
    rt.run(namespace="stocks", timeframe="yearly", cascade=False)

    # 2) Combos – stocks MQY
    print("[STEP] Combos: stocks_a_mqy_shortlist")
    rc.run(namespace="stocks", combo_name="stocks_a_mqy_shortlist")

    print("[STEP] Combos: stocks_a_mqy_all")
    rc.run(namespace="stocks", combo_name="stocks_a_mqy_all")

    print("=== [DONE] monthly_rollup ===")


PROFILES = {
    "daily_eod_stocks": profile_daily_eod_stocks,
    "daily_eod_futures": profile_daily_eod_futures,
    "weekly_rollup": profile_weekly_rollup,
    "monthly_rollup": profile_monthly_rollup,
}


# -------------------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python jobs/run_profile.py <profile_name>")
        print("Profiles:")
        for name in sorted(PROFILES):
            print(f"  - {name}")
        sys.exit(1)

    profile_name = sys.argv[1]
    if profile_name not in PROFILES:
        print(f"[ERROR] Unknown profile '{profile_name}'")
        print("Available profiles:")
        for name in sorted(PROFILES):
            print(f"  - {name}")
        sys.exit(1)

    print(f"=== Running profile '{profile_name}' at {datetime.utcnow().isoformat()}Z ===")

    # Initialize indicator engine once (for run_timeframe.apply_core)
    initialize_indicator_engine(CFG)

    fn = PROFILES[profile_name]
    fn()

    print(f"=== Finished profile '{profile_name}' at {datetime.utcnow().isoformat()}Z ===")


if __name__ == "__main__":
    main()
