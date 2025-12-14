from __future__ import annotations

# core/health.py

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from core.paths import DATA, CFG
from core import storage


@dataclass(frozen=True)
class HealthResult:
    ok: bool
    name: str
    details: str


def _norm_syms(values: Iterable[object]) -> set[str]:
    return set(
        pd.Series(list(values))
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )


def load_universe_symbols(universe: str) -> set[str]:
    """
    Supports config CSVs like:
      config/shortlist_stocks.csv  (symbol col)
      config/options_eligible_stocks.csv, etc.
    Convention: <universe>.csv
    """
    path = CFG / f"{universe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(path)
    # flexible column naming
    for col in ("symbol", "Symbol", "ticker", "Ticker"):
        if col in df.columns:
            return _norm_syms(df[col])
    raise ValueError(f"{path} must have a symbol/ticker column")


def check_parquet_nonempty(path: Path, name: str) -> HealthResult:
    if not storage.exists(path):
        return HealthResult(False, name, f"missing: {path}")
    df = storage.load_parquet(path)
    if df is None or df.empty:
        return HealthResult(False, name, f"empty: {path}")
    return HealthResult(True, name, f"ok rows={len(df)} -> {path}")


def check_combo_symbol_coverage(
    combo_path: Path,
    expected: set[str],
    name: str,
    symbol_col: str = "symbol",
) -> HealthResult:
    if not storage.exists(combo_path):
        return HealthResult(False, name, f"missing: {combo_path}")

    df = storage.load_parquet(combo_path)
    if df is None or df.empty:
        return HealthResult(False, name, f"empty: {combo_path}")

    if symbol_col not in df.columns:
        df = df.reset_index()

    actual = _norm_syms(df[symbol_col])
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing:
        return HealthResult(False, name, f"missing {len(missing)} symbols: {missing}")
    if extra:
        # not always bad, but useful to see if you expected exact equality
        return HealthResult(True, name, f"ok (has extras {len(extra)}): {extra}")

    return HealthResult(True, name, f"ok coverage: {len(expected)} symbols")


def run_combo_health(
    combos: list[str],
    universe: Optional[str] = None,
    require_nonempty: bool = True,
) -> list[HealthResult]:
    """
    combos: e.g. ["stocks_c_dwm_shortlist", "stocks_c_dwm_all"]
    universe: if provided, verify symbol coverage against config/<universe>.csv
              e.g. universe="shortlist_stocks"
    """
    results: list[HealthResult] = []

    expected: Optional[set[str]] = None
    if universe is not None:
        expected = load_universe_symbols(universe)

    for combo in combos:
        combo_path = DATA / f"combo_stocks_{combo}.parquet" if combo.startswith("c_") else DATA / f"combo_{combo}.parquet"
        # ^ if your combo filenames are always combo_stocks_<combo>.parquet, simplify to that.

        # If your naming is strictly:
        # combo_stocks_<combo>.parquet
        combo_path = DATA / f"combo_stocks_{combo}.parquet"

        if require_nonempty:
            results.append(check_parquet_nonempty(combo_path, name=f"{combo}:nonempty"))

        if expected is not None:
            results.append(
                check_combo_symbol_coverage(
                    combo_path,
                    expected=expected,
                    name=f"{combo}:coverage:{universe}",
                )
            )

    return results


def print_results(results: Iterable[HealthResult]) -> None:
    any_fail = False
    for r in results:
        if r.ok:
            print(f"[HEALTH] ✅ {r.name} — {r.details}")
        else:
            any_fail = True
            print(f"[HEALTH] ⚠️  {r.name} — {r.details}")
    if any_fail:
        print("[HEALTH] One or more checks failed.")
