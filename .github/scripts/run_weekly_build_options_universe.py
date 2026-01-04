from __future__ import annotations

# .github/scripts/run_stocks_weekly_rollup_guarded.py
# (options_eligible + symbol_to_etf_options_eligible)

import os
import subprocess
import sys
from datetime import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.health import run_combo_health, print_results
from core.guard import minutes_since_midnight, now_ny #run_guarded  # ✅ your existing core/guard.py
#from core.signal_alerts import notify_on_signals
from core.notify import notify_combo_signals

# =======================================================
# ---- Config: desired local target time + tolerance ----
# =======================================================
TARGET_TIME = time(hour=1, minute=30)  # 01:30 America/New_York
TOLERANCE_MIN = 45                      # +/- 45 minutes

# =======================================================
# ---- Options universe sanity thresholds ----
# =======================================================
# Additionally, require the new universe to be at least X% of the previous file’s size.
# If previous file is missing (first run), this check is skipped.
MIN_PCT_OF_PREV = float(os.getenv("OPTIONS_UNIVERSE_MIN_PCT_OF_PREV", "0.75"))


def _count_symbols(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    for col in ("symbol", "Symbol", "ticker", "Ticker"):
        if col in df.columns:
            return (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .nunique()
            )
    raise ValueError(f"{csv_path} must contain a symbol/ticker column")


def validate_options_universe(root: Path) -> None:
    """
    Fail loudly if options universe looks wrong.
    Checks:
      - options_eligible.csv exists and has enough symbols
      - compare to previous file count (if any)
      - mapping exists and is not tiny relative to universe
    """
    ref_dir = root / "ref"

    universe_path = ref_dir / "options_eligible.csv"
    mapping_path = ref_dir / "symbol_to_etf_options_eligible.csv"

    if not universe_path.exists():
        raise RuntimeError(f"[FATAL] Missing options universe file: {universe_path}")

    new_n = _count_symbols(universe_path)

    # Compare to previous committed file count (works because checkout is before the build;
    # after build, the file is overwritten, but git can still show HEAD^ only after commit.
    # Instead, we compare against the *pre-build* count captured earlier in run_profile().
    # We'll pass that in via env var if available; otherwise skip.
    prev_n_str = os.getenv("OPTIONS_UNIVERSE_PREV_COUNT", "").strip()
    if prev_n_str.isdigit():
        prev_n = int(prev_n_str)
        if prev_n > 0:
            pct = new_n / prev_n
            if pct < MIN_PCT_OF_PREV:
                raise RuntimeError(
                    f"[FATAL] options_eligible shrank too much: {new_n} vs prev {prev_n} "
                    f"({pct:.1%} < {MIN_PCT_OF_PREV:.0%}). Likely transient data/API issue."
                )

    # Mapping sanity (optional but useful)
    if not mapping_path.exists():
        raise RuntimeError(f"[FATAL] Missing ETF mapping file: {mapping_path}")

    map_n = _count_symbols(mapping_path)  # counts unique 'symbol' column values if present
    # mapping can be slightly smaller, but shouldn’t be *wildly* smaller
    if map_n < int(0.80 * new_n):
        raise RuntimeError(
            f"[FATAL] ETF mapping too small relative to universe: mapping {map_n}, universe {new_n}."
        )

    #print(f"[HEALTH] ✅ options_eligible OK — {new_n} symbols (mapping ~{map_n})")
    print(
        f"[HEALTH] ✅ options_eligible OK — "
        f"{new_n} symbols ({new_n / prev_n:.0%} of previous)"
        if prev_n_str.isdigit()
        else f"[HEALTH] ✅ options_eligible OK — {new_n} symbols"
    )



def run_profile() -> None:
    root = Path(__file__).resolve().parents[2]

    # Capture previous count BEFORE rebuild (so we can compare shrinkage)
    prev_path = root / "ref" / "options_eligible.csv"
    if prev_path.exists():
        try:
            os.environ["OPTIONS_UNIVERSE_PREV_COUNT"] = str(_count_symbols(prev_path))
        except Exception:
            # If previous file is malformed, just skip pct check
            os.environ["OPTIONS_UNIVERSE_PREV_COUNT"] = ""
    else:
        os.environ["OPTIONS_UNIVERSE_PREV_COUNT"] = ""
    
    cmds = [
        # ---------------------------------------------------------
        # 1) Refresh universes
        # ---------------------------------------------------------
        [sys.executable, str(root / "jobs" / "run_build_options_universe.py")],
    ]

    for cmd in cmds:
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # ✅ Sanity-check outputs (fail loudly if bad)
    validate_options_universe(root)



def main() -> None:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")

    # ✅ Manual runs always execute, regardless of time
    if event_name == "workflow_dispatch":
        print("[INFO] Triggered via workflow_dispatch; bypassing time-window guard.")
        run_profile()
        print("[OK] options universe build completed.")
        return

    # ✅ Scheduled runs: enforce DST-aware time window
    now_time = now_ny()
    #weekday = now.weekday()  # Monday=0 ... Sunday=6
    weekday = now_time.weekday()  # Monday=0 ... Sunday=6

    # Require Sunday (6)
    if weekday != 6:
        print(f"[INFO] Today ({today}) is not Sunday in NY. Skipping options-universe build.")
        sys.exit(0)

    now_min = minutes_since_midnight(now_time)
    target_min = minutes_since_midnight(TARGET_TIME)
    diff = abs(now_min - target_min)

    if diff > TOLERANCE_MIN:
        print(
            f"[INFO] {now_time} local (NY) is outside +/-{TOLERANCE_MIN} minutes "
            f"of target {TARGET_TIME}. Skipping options-universe build."
        )
        sys.exit(0)

    print(f"[INFO] Within window at {now_time} NY. Running weekly stocks rollup profile...")
    run_profile()
    print("[OK] options universe build completed.")


if __name__ == "__main__":
    main()
