from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diagnostics.outcomes.coverage import (
    audit_horizon,
    corporate_action_events,
    revision_measures,
    summarize_coverage,
)
from diagnostics.outcomes.price_source import CachedPriceSource, RollingDailyPriceSource
from diagnostics.outcomes.specs import CoverageStatus, Direction, HORIZON_SPECS


def frame(start="2026-01-01", end="2026-02-15", close=110.0):
    index = pd.bdate_range(start, end)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 5,
            "low": close - 5,
            "close": close,
            "adj_close": close,
        },
        index=index,
    )


def audit(prices, *, asof=date(2026, 2, 15), horizon=None):
    return audit_horizon(
        direction=Direction.LONG,
        entry_date=date(2026, 1, 1),
        immutable_entry_close=100,
        horizon=horizon or HORIZON_SPECS["stocks_c_dwm_all"].horizons[0],
        frame=prices,
        dataset_asof=asof,
    )


def test_missing_symbol_history_is_not_generic_missing():
    result = audit(None)
    assert result["coverage_status"] == CoverageStatus.MISSING_SYMBOL_HISTORY.value
    assert not result["terminal_covered"] and not result["path_covered"]


def test_entry_predates_retained_history_when_terminal_also_rolled_off():
    result = audit(frame("2026-02-10", "2026-02-15"))
    assert (
        result["coverage_status"]
        == CoverageStatus.ENTRY_PREDATES_RETAINED_HISTORY.value
    )
    assert not result["terminal_covered"]


def test_terminal_coverage_is_separate_from_incomplete_path_coverage():
    prices = frame("2026-01-20", "2026-02-15")
    result = audit(prices)
    assert result["terminal_covered"]
    assert not result["path_covered"]
    assert result["coverage_status"] == CoverageStatus.INCOMPLETE_EXCURSION_WINDOW.value


def test_forward_resolution_and_immutable_entry_close_are_preserved():
    prices = frame()
    prices.loc[pd.Timestamp("2026-01-01"), "close"] = 999
    prices.loc[pd.Timestamp("2026-02-02"), "close"] = 110
    result = audit(prices)
    assert result["resolved_exit_date"] == date(2026, 2, 2)  # Jan 31 weekend
    assert result["resolved_exit_date"] >= result["target_calendar_date"]
    assert result["directional_return"] == pytest.approx(0.10)
    assert result["terminal_covered"] and result["path_covered"]


def test_invalid_path_data_preserves_terminal_coverage():
    prices = frame()
    prices.loc[pd.Timestamp("2026-01-15"), "high"] = np.nan
    result = audit(prices)
    assert result["terminal_covered"] and not result["path_covered"]
    assert result["coverage_status"] == CoverageStatus.INVALID_PRICE_DATA.value


def test_mature_and_immature_are_distinct_including_mqy():
    mqy = HORIZON_SPECS["stocks_a_mqy_all"].horizons[-1]
    result = audit(frame(), asof=date(2026, 9, 1), horizon=mqy)
    assert result["coverage_status"] == CoverageStatus.IMMATURE.value
    assert not result["theoretically_mature"]


def test_revision_bucket_calculations_are_strict_descriptive_thresholds():
    exact = revision_measures(100, 100)
    changed = revision_measures(100, 101.01)
    assert exact["exact_or_near_exact"] and not exact["over_1bp"]
    assert changed["over_1bp"] and changed["over_10bp"]
    assert changed["over_50bp"] and changed["over_1pct"]
    assert not changed["over_5pct"]
    assert changed["absolute_difference"] == pytest.approx(1.01)


def test_corporate_action_ratio_break_and_split_like_heuristic():
    prices = pd.DataFrame(
        {"close": [100.0, 50.0], "adj_close": [50.0, 50.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )
    events = corporate_action_events(prices)
    assert len(events) == 1
    assert bool(events.iloc[0].ratio_break_flag)
    assert bool(events.iloc[0].split_like_flag)
    assert corporate_action_events(prices.drop(columns="adj_close")).empty


def test_cached_source_loads_each_symbol_once():
    calls = []

    def reader(key):
        calls.append(key)
        return frame()

    source = CachedPriceSource(RollingDailyPriceSource(reader, prefix="data"))
    first = source.load("abc")
    second = source.load("ABC")
    assert first is second
    assert calls == ["data/bars/stocks_daily/ABC.parquet"]


def test_coverage_summary_uses_theoretically_mature_denominator():
    observations = pd.DataFrame(
        {
            "combo": ["x"] * 3,
            "direction": ["LONG"] * 3,
            "theoretically_mature": [True, True, False],
            "terminal_covered": [True, False, False],
            "path_covered": [True, False, False],
            "coverage_status": [
                CoverageStatus.MATURE.value,
                CoverageStatus.MISSING_SYMBOL_HISTORY.value,
                CoverageStatus.IMMATURE.value,
            ],
            "corporate_action_flag": [False, False, False],
        }
    )
    result = summarize_coverage(observations, ["combo", "direction"]).iloc[0]
    assert result.theoretically_mature_count == 2
    assert result.terminal_coverage_of_theoretically_mature == 0.5
    assert result.path_coverage_of_theoretically_mature == 0.5
    assert result.immature_count == 1 and result.missing_symbol_count == 1


def test_phase4a_audit_source_has_no_mutation_or_external_provider_calls():
    source = Path("diagnostics/analyze_stock_options_phase4a_coverage.py").read_text()
    assert "put_object" not in source
    assert "upload_file" not in source
    assert "delete_object" not in source
    assert "yfinance" not in source and "yf." not in source
    assert "get_object" in source
