from __future__ import annotations

# core/health.py

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from core.paths import DATA, CFG
from core import storage


# -------------------------
# models
# -------------------------

@dataclass(frozen=True)
class HealthResult:
    ok: bool
    name: str
    details: str


# -------------------------
# helpers
# -------------------------

def _norm_syms(values: Iterable[object]) -> set[str]:
    return set(
        pd.Series(list(values))
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )


def load_universe_symbols(universe_csv: str) -> set[str]:
    """
    universe_csv: e.g. "shortlist_stocks.csv"
    """
    path = CFG / universe_csv
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(path)
    for col in ("symbol", "Symbol", "ticker", "Ticker"):
        if col in df.columns:
            return _norm_syms(df[col])

    raise ValueError(f"{path} must contain a symbol/ticker column")


# -------------------------
# checks
# -------------------------

def check_combo_nonempty(combo_name: str) -> HealthResult:
    path = DATA / f"combo_{combo_name}.parquet"

    if not storage.exists(path):
        return HealthResult(False, combo_name, f"missing file: {path}")

    df = storage.load_parquet(path)
    if df is None or df.empty:
        return HealthResult(False, combo_name, "combo parquet is empty")

    return HealthResult(True, combo_name, f"ok rows={len(df)}")


def check_combo_symbol_coverage(
    combo_name: str,
    expected_symbols: set[str],
    symbol_col: str = "symbol",
) -> HealthResult:
    path = DATA / f"combo_{combo_name}.parquet"

    if not storage.exists(path):
        return HealthResult(False, combo_name, "missing file")

    df = storage.load_parquet(path)
    if df is None or df.empty:
        return HealthResult(False, combo_name, "empty combo")

    if symbol_col not in df.columns:
        df = df.reset_index()

    actual = _norm_syms(df[symbol_col])
    missing = sorted(expected_symbols - actual)

    if missing:
        return HealthResult(
            False,
            combo_name,
            f"missing {len(missing)} symbols: {missing}",
        )

    return HealthResult(
        True,
        combo_name,
        f"ok coverage ({len(expected_symbols)} symbols)",
    )


# -------------------------
# runner
# -------------------------

def run_combo_health(
    combos: list[str],
    universe_csv: Optional[str] = None,
    require_nonempty: bool = True,
) -> list[HealthResult]:
    """
    combos:
        [
          "stocks_c_dwm_shortlist",
          "futures_3_dwm_shortlist",
          ...
        ]

    universe_csv:
        "shortlist_stocks.csv" (optional)
    """
    results: list[HealthResult] = []

    expected: Optional[set[str]] = None
    if universe_csv:
        expected = load_universe_symbols(universe_csv)

    for combo in combos:
        if require_nonempty:
            results.append(check_combo_nonempty(combo))

        if expected is not None:
            results.append(check_combo_symbol_coverage(combo, expected))

    return results

"""
def print_results(results: Iterable[HealthResult]) -> None:
    failed = False
    for r in results:
        if r.ok:
            print(f"[HEALTH] ✅ {r.name} — {r.details}")
        else:
            failed = True
            print(f"[HEALTH] ⚠️  {r.name} — {r.details}")

    if failed:
        print("[HEALTH] One or more checks failed.")
"""

def print_results(results: Iterable[HealthResult], fail_on_error: bool = True) -> None:
    failed = False
    for r in results:
        if r.ok:
            print(f"[HEALTH] ✅ {r.name} — {r.details}")
        else:
            failed = True
            print(f"[HEALTH] ⚠️  {r.name} — {r.details}")

    if failed:
        msg = "[HEALTH] One or more checks failed."
        print(msg)
        if fail_on_error:
            raise RuntimeError(msg)
