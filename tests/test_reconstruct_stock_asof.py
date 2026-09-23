from __future__ import annotations

import inspect
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import reconstruct_stock_asof as diagnostic


def daily_fixture() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", "2026-09-23")
    values = np.arange(len(index), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "Open": values,
            "High": values + 2,
            "Low": values - 1,
            "Close": values + 1,
            "Adj Close": values + 0.5,
            "Volume": values * 10,
        },
        index=index,
    )


def test_strict_asof_and_partial_monday_week_and_month():
    cutoff = date(2026, 9, 21)
    daily = diagnostic.normalize_provider_daily(daily_fixture(), cutoff)
    weekly = diagnostic.aggregate_partial_bars(daily, "weekly")
    monthly = diagnostic.aggregate_partial_bars(daily, "monthly")

    assert daily.index.max() == pd.Timestamp("2026-09-21")
    assert not (daily.index.date > cutoff).any()
    pd.testing.assert_series_equal(
        weekly.iloc[-1], daily.iloc[-1], check_names=False, check_dtype=False
    )
    september = daily.loc["2026-09"]
    assert monthly.index[-1] == pd.Timestamp("2026-09-01")
    assert monthly.iloc[-1]["open"] == september.iloc[0]["open"]
    assert monthly.iloc[-1]["close"] == september.iloc[-1]["close"]
    assert monthly.iloc[-1]["volume"] == september["volume"].sum()


def test_reconstruct_has_warmup_preserves_close_and_never_requests_future():
    calls = []

    def fetcher(symbol, start, end):
        calls.append((symbol, start, end))
        return daily_fixture()

    bars, summary, metadata = diagnostic.reconstruct(
        "CRWD",
        date(2026, 9, 21),
        Path("config/indicator_params.yaml"),
        fetcher=fetcher,
    )
    assert calls[0][2] == date(2026, 9, 22)
    assert len(bars["daily"]) >= 126
    assert len(bars["weekly"]) >= 104
    assert len(bars["monthly"]) >= 24
    assert metadata["post_cutoff_observations"] == 0
    assert metadata["exact_historical_production_evidence"] is False
    assert (
        summary.set_index("timeframe").loc["daily", "close"]
        == bars["daily"].iloc[-1].close
    )
    assert (
        summary.set_index("timeframe").loc["weekly", "close"]
        == bars["weekly"].iloc[-1].close
    )


def test_comparison_distinguishes_null_from_missing_evidence():
    summary = pd.DataFrame(
        [
            {
                "timeframe": timeframe,
                "timestamp": "2026-09-21",
                "observation_count": 200,
                **{field: 1.0 for field in diagnostic.OHLCV},
                **{field: 0.0 for field in diagnostic.MATERIAL_INDICATORS},
            }
            for timeframe in ("daily", "weekly", "monthly")
        ]
    )
    persisted = pd.Series({"lower_close": np.nan, "middle_close": np.nan})
    compared = diagnostic.compare_persisted_row(summary, persisted)
    lower_close = compared.query("persisted_column == 'lower_close'").iloc[0]
    upper_close = compared.query("persisted_column == 'upper_close'").iloc[0]
    assert lower_close.comparison == "mismatch"
    assert pd.isna(lower_close.persisted)
    assert upper_close.comparison == "unavailable"


def test_module_is_structurally_read_only():
    source = inspect.getsource(diagnostic)
    forbidden = (
        "save_parquet(",
        "put_object(",
        "delete_object(",
        "jobs.run_timeframe",
        "jobs.run_combo",
        "core.storage",
    )
    assert all(token not in source for token in forbidden)


def test_cli_records_provider_unavailability_without_traceback(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("provider blocked")

    monkeypatch.setattr(diagnostic, "reconstruct", unavailable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconstruct_stock_asof.py",
            "--output-dir",
            str(tmp_path),
        ],
    )
    diagnostic.main()
    metadata = json.loads((tmp_path / "reconstruction_metadata.json").read_text())
    assert metadata["status"] == "unavailable"
    assert metadata["exact_historical_production_evidence"] is False
