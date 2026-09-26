from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from etl.eod_family import acquire_stock_eod_family
from jobs import run_stock_eod_family as runner
from jobs import run_timeframe


SESSION = pd.Timestamp("2026-09-23")


def frame(interval: str, close=100.0, *, last_trade=True):
    result = pd.DataFrame(
        {"open": [99.0], "high": [101.0], "low": [98.0], "close": [close],
         "adj_close": [99.5], "volume": [1000.0]},
        index=[SESSION],
    )
    metadata = {"priceHint": 2}
    if interval != "1d" and last_trade:
        metadata["lastTrade"] = {"Price": close, "Time": SESSION}
    result.attrs["eod_provenance"] = {
        "source_interval": interval,
        "adjustment_mode": "raw_unadjusted",
        "session_semantics": "regular",
        "provider_metadata": metadata,
    }
    return result


def test_process_preloaded_uses_existing_path_and_never_fetches(monkeypatch, tmp_path):
    existing = frame("1d", 99.0)
    new = frame("1d", 100.0)
    target = tmp_path / "bars/stocks_daily/AAPL.parquet"
    saved = {}
    indicator_inputs = []

    monkeypatch.setattr(run_timeframe, "parquet_path", lambda *args: target)
    monkeypatch.setattr(run_timeframe.storage, "exists", lambda path: path == target)
    monkeypatch.setattr(run_timeframe.storage, "load_parquet", lambda path: existing.copy())
    monkeypatch.setattr(run_timeframe.storage, "save_parquet",
                        lambda value, path: saved.update({path: value.copy()}))
    monkeypatch.setattr(run_timeframe, "safe_load_eod",
                        lambda *args, **kwargs: pytest.fail("provider called"))
    monkeypatch.setattr(
        run_timeframe, "apply_core",
        lambda value, **kwargs: indicator_inputs.append(value.copy()) or value,
    )

    outcome = run_timeframe.process_one_preloaded(
        "stocks", "daily", "AAPL", new, 260,
    )

    assert outcome.accepted
    assert indicator_inputs[0].iloc[-1].close == 100.0
    assert saved[target].iloc[-1].close == 100.0


def test_process_preloaded_rejects_malformed_before_indicators_and_preserves_history(
    monkeypatch, tmp_path
):
    existing = frame("1wk", 99.0)
    malformed = frame("1wk", np.nan)
    target = tmp_path / "bars/stocks_weekly/AAPL.parquet"
    saves = []

    monkeypatch.setattr(run_timeframe, "parquet_path", lambda *args: target)
    monkeypatch.setattr(run_timeframe.storage, "exists", lambda path: True)
    monkeypatch.setattr(run_timeframe.storage, "load_parquet", lambda path: existing.copy())
    monkeypatch.setattr(run_timeframe.storage, "save_parquet",
                        lambda *args: saves.append(args))
    monkeypatch.setattr(run_timeframe, "apply_core",
                        lambda *args, **kwargs: pytest.fail("indicators called"))

    outcome = run_timeframe.process_one_preloaded(
        "stocks", "weekly", "AAPL", malformed, 210,
    )
    assert not outcome.accepted
    assert not saves
    assert existing.iloc[-1].close == 99.0


def test_snapshot_excludes_rejected_symbol_even_when_stale_rolling_exists(
    monkeypatch, tmp_path
):
    stored = {symbol: frame("1d", value) for symbol, value in (("AAPL", 100), ("MSFT", 200))}
    writes = []
    monkeypatch.setattr(run_timeframe, "parquet_path",
                        lambda root, timeframe, symbol: tmp_path / f"{symbol}.parquet")
    monkeypatch.setattr(run_timeframe.storage, "exists", lambda path: True)
    monkeypatch.setattr(run_timeframe.storage, "load_parquet",
                        lambda path: stored[path.stem].copy())
    monkeypatch.setattr(run_timeframe.storage, "save_parquet",
                        lambda value, path: writes.append((value.copy(), path)))
    monkeypatch.setattr(run_timeframe, "get_snapshot_base_cols",
                        lambda *args: list(stored["AAPL"].columns))

    snapshot = run_timeframe.build_timeframe_snapshot(
        "stocks", "daily", ["AAPL", "MSFT"], {"MSFT"},
    )
    assert snapshot.symbol.tolist() == ["AAPL"]
    assert writes[0][1] == run_timeframe.DATA / "snapshot_stocks_daily.parquet"


def test_standalone_ingest_delegates_to_preloaded_processing(monkeypatch):
    calls = []
    monkeypatch.setattr(run_timeframe, "safe_load_eod", lambda *args, **kwargs: frame("1d"))
    monkeypatch.setattr(
        run_timeframe, "process_one_preloaded",
        lambda *args, **kwargs: calls.append((args, kwargs)) or run_timeframe.ProcessOutcome(True),
    )
    monkeypatch.setattr(run_timeframe, "write_eod_reconciliation_telemetry", lambda *args: None)
    assert not run_timeframe.ingest_one("stocks", "daily", ["AAPL"], "regular", 260)
    assert len(calls) == 1 and calls[0][0][2] == "AAPL"


def test_family_universe_mismatch_fails_before_acquisition(monkeypatch):
    universes = {"daily": ["AAPL"], "weekly": ["AAPL", "MSFT"], "monthly": ["AAPL"]}
    monkeypatch.setattr(runner, "symbols_for_timeframe",
                        lambda namespace, timeframe: universes[timeframe])
    called = []
    with pytest.raises(RuntimeError, match="universe mismatch"):
        runner.run_family(run_id="run", acquire=lambda *args, **kwargs: called.append(args))
    assert not called


def execute_family(
    monkeypatch, closes, *, failing_timeframe=None, accounting_delta=0,
    reconciliation_request_delta=0,
):
    calls, processed, snapshots, writes = [], [], [], []
    monkeypatch.setattr(runner, "symbols_for_timeframe", lambda *args: ["AAPL"])
    monkeypatch.setattr(runner, "initialize_indicator_engine", lambda *args: None)
    monkeypatch.setattr(
        runner, "capture_current_pointer_state",
        lambda *args: SimpleNamespace(revision=None, previous_family_run_id=None),
    )
    monkeypatch.setattr(runner.storage, "save_parquet",
                        lambda value, path: writes.append((value.copy(), path)))
    monkeypatch.setattr(
        runner, "process_one_preloaded",
        lambda namespace, timeframe, symbol, value, window, **kwargs:
        processed.append((timeframe, value.copy(), window)) or run_timeframe.ProcessOutcome(True),
    )
    monkeypatch.setattr(
        runner, "build_timeframe_snapshot_frame",
        lambda namespace, timeframe, symbols, rejected:
        snapshots.append((timeframe, set(rejected))) or pd.DataFrame({"symbol": ["AAPL"]}),
    )
    monkeypatch.setattr(
        runner, "publish_stock_eod_family",
        lambda run_id, *args, **kwargs: SimpleNamespace(
            pointer=SimpleNamespace(family_run_id=run_id, previous_family_run_id=None)
        ),
    )
    monkeypatch.setattr(runner, "write_legacy_timeframe_snapshot", lambda *args: None)

    def loader(symbol, timeframe, **kwargs):
        calls.append((timeframe, kwargs["window_bars"]))
        if timeframe == failing_timeframe:
            raise RuntimeError("unavailable")
        interval = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe]
        return frame(interval, closes[timeframe])

    reference_calls = []

    def reference_loader(symbol, timeframe, **kwargs):
        reference_calls.append((symbol, timeframe, kwargs["window_bars"]))
        interval = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe]
        return frame(interval, 100.0)

    def acquire(run_id, symbol, **kwargs):
        family = acquire_stock_eod_family(run_id, symbol, loader=loader, **kwargs)
        if accounting_delta:
            object.__setattr__(family.requests, "attempted", family.requests.attempted + accounting_delta)
        if reconciliation_request_delta:
            object.__setattr__(
                family.requests, "reconciliation_provider_requests",
                family.requests.reconciliation_provider_requests + reconciliation_request_delta,
            )
        return family

    path = runner.run_family(
        run_id="prod-20260926T000000Z-00000001", acquire=acquire, reference_loader=reference_loader,
    )
    return path, calls, processed, snapshots, writes, reference_calls


def test_valid_family_uses_configured_windows_exactly_once_and_writes_telemetry(monkeypatch):
    path, calls, processed, snapshots, writes, reference_calls = execute_family(
        monkeypatch, {"daily": 100.0, "weekly": 100.0, "monthly": 100.0}
    )
    expected_windows = runner.production_window_bars()
    assert calls == [(tf, expected_windows[interval]) for interval, tf in zip(runner.FAMILY_INTERVALS, runner.TIMEFRAMES)]
    assert [item[0] for item in processed] == list(runner.TIMEFRAMES)
    assert all(not rejected for _, rejected in snapshots)
    assert path == runner.DATA / "_health/stocks_eod_family/prod-20260926T000000Z-00000001/manifest.parquet"
    telemetry = writes[0][0]
    summary = telemetry.loc[telemetry.record_type == "run_summary"].iloc[0]
    assert summary.requests_attempted == 3
    assert summary.primary_provider_attempts == 3
    assert summary.reference_provider_attempts == 6
    assert summary.total_provider_attempts == 9
    assert summary.reconciliation_provider_requests == 0
    assert summary.downstream_provider_requests == 0
    assert len(reference_calls) == 6
    assert {call[2] for call in reference_calls} == {runner.REFERENCE_EOD_WINDOW_BARS}
    assert len(telemetry.loc[telemetry.record_type == "reference"]) == 6
    assert set(telemetry.loc[telemetry.record_type == "interval", "processing_status"]) == {"accepted"}


@pytest.mark.parametrize(
    ("closes", "expected_processed", "daily_close"),
    [
        ({"daily": np.nan, "weekly": 100.0, "monthly": 100.0}, {"daily", "weekly", "monthly"}, 100.0),
        ({"daily": np.nan, "weekly": np.nan, "monthly": 100.0}, {"daily", "monthly"}, 100.0),
        ({"daily": np.nan, "weekly": 100.0, "monthly": 102.0}, {"weekly", "monthly"}, None),
        ({"daily": 100.0, "weekly": np.nan, "monthly": 100.0}, {"daily", "monthly"}, None),
        ({"daily": 100.0, "weekly": 100.0, "monthly": np.nan}, {"daily", "weekly"}, None),
    ],
)
def test_family_finalization_and_partial_exclusion(
    monkeypatch, closes, expected_processed, daily_close
):
    _, _, processed, snapshots, _, _ = execute_family(monkeypatch, closes)
    assert {item[0] for item in processed} == expected_processed
    if daily_close is not None:
        daily = next(value for timeframe, value, _ in processed if timeframe == "daily")
        assert daily.iloc[-1].close == daily_close
    rejected = dict(snapshots)
    for timeframe in runner.TIMEFRAMES:
        assert rejected[timeframe] == (set() if timeframe in expected_processed else {"AAPL"})


def test_provider_exception_preserves_other_intervals(monkeypatch):
    _, calls, processed, snapshots, _, _ = execute_family(
        monkeypatch,
        {"daily": 100.0, "weekly": 100.0, "monthly": 100.0},
        failing_timeframe="weekly",
    )
    assert [call[0] for call in calls] == list(runner.TIMEFRAMES)
    assert {item[0] for item in processed} == {"daily", "monthly"}
    assert dict(snapshots)["weekly"] == {"AAPL"}


def test_request_accounting_violation_fails_before_snapshots(monkeypatch):
    with pytest.raises(RuntimeError, match="request invariant"):
        execute_family(
            monkeypatch,
            {"daily": 100.0, "weekly": 100.0, "monthly": 100.0},
            accounting_delta=1,
        )


def test_reconciliation_request_invariant_violation_fails(monkeypatch):
    with pytest.raises(RuntimeError, match="reconciliation request invariant"):
        execute_family(
            monkeypatch,
            {"daily": 100.0, "weekly": 100.0, "monthly": 100.0},
            reconciliation_request_delta=1,
        )


def test_all_unavailable_required_timeframe_fails_snapshot(monkeypatch):
    monkeypatch.setattr(runner, "symbols_for_timeframe", lambda *args: ["AAPL"])
    monkeypatch.setattr(runner, "initialize_indicator_engine", lambda *args: None)
    monkeypatch.setattr(runner.storage, "save_parquet", lambda *args: None)
    monkeypatch.setattr(runner, "process_one_preloaded",
                        lambda *args, **kwargs: run_timeframe.ProcessOutcome(True))

    def acquire(run_id, symbol, **kwargs):
        return acquire_stock_eod_family(
            run_id, symbol, loader=lambda *args, **kwargs: None, **kwargs,
        )

    def reference_loader(symbol, timeframe, **kwargs):
        interval = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe]
        return frame(interval)

    with pytest.raises(RuntimeError, match="no valid rows"):
        runner.run_family(
            run_id="run", acquire=acquire, reference_loader=reference_loader,
        )


def test_reference_failure_stops_before_primary_acquisition(monkeypatch):
    monkeypatch.setattr(runner, "symbols_for_timeframe", lambda *args: ["AAPL"])
    monkeypatch.setattr(runner, "initialize_indicator_engine", lambda *args: None)
    primary_calls = []
    with pytest.raises(RuntimeError, match="reference-data acquisition failed"):
        runner.run_family(
            run_id="run",
            acquire=lambda *args, **kwargs: primary_calls.append(args),
            reference_loader=lambda *args, **kwargs: None,
        )
    assert not primary_calls


def test_compatibility_mirrors_follow_commit_and_failure_is_explicit(monkeypatch, capsys):
    events = []
    monkeypatch.setattr(runner, "symbols_for_timeframe", lambda *args: ["AAPL"])
    monkeypatch.setattr(runner, "initialize_indicator_engine", lambda *args: None)
    monkeypatch.setattr(
        runner, "capture_current_pointer_state",
        lambda *args: SimpleNamespace(revision=None, previous_family_run_id=None),
    )
    monkeypatch.setattr(runner.storage, "save_parquet", lambda *args: None)
    monkeypatch.setattr(
        runner, "process_one_preloaded",
        lambda *args, **kwargs: run_timeframe.ProcessOutcome(True),
    )
    monkeypatch.setattr(
        runner, "build_timeframe_snapshot_frame",
        lambda *args, **kwargs: pd.DataFrame({"symbol": ["AAPL"]}),
    )
    monkeypatch.setattr(
        runner, "publish_stock_eod_family",
        lambda run_id, *args, **kwargs: events.append("commit") or SimpleNamespace(
            pointer=SimpleNamespace(family_run_id=run_id, previous_family_run_id=None)
        ),
    )

    def mirror(*args):
        events.append(f"mirror:{args[1]}")
        raise OSError("mirror failed")

    monkeypatch.setattr(runner, "write_legacy_timeframe_snapshot", mirror)

    def acquire(run_id, symbol, **kwargs):
        return acquire_stock_eod_family(
            run_id, symbol,
            loader=lambda symbol, timeframe, **kwargs: frame(
                {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe]
            ), **kwargs,
        )

    with pytest.raises(OSError, match="mirror failed"):
        runner.run_family(
            run_id="prod-20260926T110000Z-0000000d", acquire=acquire,
            reference_loader=lambda symbol, timeframe, **kwargs: frame(
                {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe]
            ),
        )
    assert events == ["commit", "mirror:daily"]
    assert "COMMITTED_BUT_MIRROR_FAILED" in capsys.readouterr().out
