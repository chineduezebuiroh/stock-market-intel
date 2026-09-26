from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etl.eod_family import acquire_stock_eod_family
from indicators import composite_spy_qqq_volume_ma_ratio as benchmark_indicator
from jobs import run_stock_eod_family as runner
from jobs import run_timeframe


SESSION = pd.Timestamp("2026-09-23")


def history(interval: str, close: float = 100.0, *, malformed_terminal=False, periods=40):
    index = pd.bdate_range(end=SESSION, periods=periods)
    closes = np.linspace(close - 5, close, periods)
    if malformed_terminal:
        closes[-1] = np.nan
    result = pd.DataFrame(
        {
            "open": np.linspace(close - 5.5, close - 1, periods),
            "high": np.linspace(close - 4, close + 1, periods),
            "low": np.linspace(close - 6, close - 2, periods),
            "close": closes,
            "adj_close": closes.copy(),
            "volume": np.linspace(1_000, 2_000, periods),
        },
        index=index,
    )
    metadata = {"priceHint": 2}
    if interval != "1d":
        metadata["lastTrade"] = {"Price": close, "Time": SESSION}
    result.attrs["eod_provenance"] = {
        "source_interval": interval,
        "adjustment_mode": "raw_unadjusted",
        "session_semantics": "regular",
        "provider_metadata": metadata,
    }
    return result


def test_authoritative_family_handoff_is_provider_closed_and_publishes_finalized_frames(
    monkeypatch, tmp_path
):
    symbols = ["AAPL", "MSFT"]
    primary_calls = []
    reference_calls = []
    storage_state: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(runner, "DATA", tmp_path)
    monkeypatch.setattr(run_timeframe, "DATA", tmp_path)
    monkeypatch.setattr(runner, "symbols_for_timeframe", lambda *args: symbols)

    def exists(path):
        return str(path) in storage_state

    def load(path):
        return storage_state[str(path)].copy(deep=True)

    def save(value, path):
        storage_state[str(path)] = value.copy(deep=True)

    monkeypatch.setattr(runner.storage, "exists", exists)
    monkeypatch.setattr(runner.storage, "load_parquet", load)
    monkeypatch.setattr(runner.storage, "save_parquet", save)

    old_weekly = history("1wk", 90.0)
    msft_weekly_path = tmp_path / "bars/stocks_weekly/MSFT.parquet"
    storage_state[str(msft_weekly_path)] = old_weekly.copy(deep=True)

    def provider(symbol, timeframe, **kwargs):
        primary_calls.append((symbol, timeframe, kwargs["window_bars"]))
        interval = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe]
        if symbol == "AAPL" and timeframe == "daily":
            return history(interval, 100.0, malformed_terminal=True)
        if symbol == "MSFT" and timeframe == "weekly":
            return history(interval, 200.0, malformed_terminal=True)
        return history(interval, 100.0 if symbol == "AAPL" else 200.0)

    def acquire(run_id, symbol, **kwargs):
        return acquire_stock_eod_family(run_id, symbol, loader=provider, **kwargs)

    def reference_provider(symbol, timeframe, **kwargs):
        reference_calls.append((symbol, timeframe, kwargs["window_bars"]))
        interval = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe]
        return history(interval, 400.0, periods=runner.REFERENCE_EOD_WINDOW_BARS)

    # This is the actual direct provider boundary used by the benchmark indicator.
    # Once explicit preloading has completed, any fallback call is a test failure.
    monkeypatch.setattr(
        benchmark_indicator,
        "load_eod",
        lambda *args, **kwargs: pytest.fail("hidden downstream provider call"),
    )

    telemetry_path = runner.run_family(
        run_id="prod-20260926T010000Z-00000002",
        acquire=acquire,
        reference_loader=reference_provider,
    )

    assert len(primary_calls) == 3 * len(symbols)
    assert len(reference_calls) == 6
    assert {call[:2] for call in reference_calls} == {
        (symbol, timeframe)
        for symbol in runner.REFERENCE_SYMBOLS
        for timeframe in runner.TIMEFRAMES
    }

    aapl_daily = storage_state[str(tmp_path / "bars/stocks_daily/AAPL.parquet")]
    assert aapl_daily.iloc[-1].close == 100.0
    assert "spy_qqq_vol_ma_ratio" in aapl_daily.columns

    aapl_weekly = storage_state[str(tmp_path / "bars/stocks_weekly/AAPL.parquet")]
    aapl_monthly = storage_state[str(tmp_path / "bars/stocks_monthly/AAPL.parquet")]
    assert aapl_weekly.iloc[-1].close == 100.0
    assert aapl_monthly.iloc[-1].close == 100.0

    pd.testing.assert_frame_equal(storage_state[str(msft_weekly_path)], old_weekly)
    daily_snapshot = storage_state[str(tmp_path / "snapshot_stocks_daily.parquet")]
    weekly_snapshot = storage_state[str(tmp_path / "snapshot_stocks_weekly.parquet")]
    monthly_snapshot = storage_state[str(tmp_path / "snapshot_stocks_monthly.parquet")]
    assert set(daily_snapshot.symbol) == set(symbols)
    assert set(monthly_snapshot.symbol) == set(symbols)
    assert set(weekly_snapshot.symbol) == {"AAPL"}
    assert daily_snapshot.loc[daily_snapshot.symbol == "AAPL", "close"].iloc[0] == 100.0

    telemetry = storage_state[str(telemetry_path)]
    summary = telemetry.loc[telemetry.record_type == "run_summary"].iloc[0]
    assert summary.primary_provider_attempts == 6
    assert summary.reference_provider_attempts == 6
    assert summary.total_provider_attempts == 12
    assert summary.reconciliation_provider_requests == 0
    assert summary.downstream_provider_requests == 0
