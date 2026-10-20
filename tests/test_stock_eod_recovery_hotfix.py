"""Offline regressions for prepublication coverage and mandatory ETF ingestion."""
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core import storage, guard
from core.stock_eod_family import publish_stock_eod_family, current_pointer_path
from etl.window import update_fixed_window
from indicators.helpers import _atr
from jobs import run_etf_trends as etf, run_stock_eod_family as family, run_timeframe


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network forbidden in recovery tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(storage, "_DATA_BACKEND", "local")
    monkeypatch.setenv("STOCK_EOD_CAPTURE_APPLICATION", "0")


def bars():
    close = np.arange(40, dtype=float) + 100
    return pd.DataFrame({"open": close, "high": close + 2, "low": close - 2,
                         "close": close, "adj_close": close, "volume": close * 10},
                        index=pd.date_range("2026-08-01", periods=40, name="date"))


def provenance():
    return {"provider_metadata": {
        "regularMarketTime": pd.Timestamp("2026-10-19", tz="UTC"),
        "lastTrade": {"Time": pd.Timestamp("2026-10-19"), "Price": 100.},
        "currentTradingPeriod": {"regular": {"start": pd.Timestamp("2026-10-19")}},
        "tradingPeriods": pd.DataFrame({"start": [pd.Timestamp("2026-10-19")]})}}


@pytest.mark.parametrize("timeframe", ["daily", "weekly"])
@pytest.mark.parametrize("has_existing", [False, True])
def test_etf_canonical_boundary_roundtrip_and_atr(monkeypatch, tmp_path, timeframe, has_existing, capsys):
    incoming = bars(); incoming.attrs = {"eod_provenance": provenance(), "units": "USD"}
    existing = bars().iloc[:-1].copy() if has_existing else pd.DataFrame()
    existing.attrs = {"eod_provenance": provenance(), "source_note": "durable"}
    # Original no-history branch preserves the unsafe attrs and fails exactly
    # as production did. DataFrame metadata also breaks ATR concat comparisons.
    unsafe = update_fixed_window(incoming, pd.DataFrame(), 40)
    with pytest.warns(UserWarning, match="Could not serialize"):
        with pytest.raises(TypeError, match="Timestamp.*JSON serializable"):
            unsafe.to_parquet(tmp_path / "unsafe.parquet")
    with pytest.raises(ValueError, match="truth value of a DataFrame"):
        _atr(unsafe, 5)

    target = tmp_path / "rolling.parquet"
    if has_existing:
        # Inject historical acquisition attrs in memory: legacy parquet cannot
        # necessarily serialize them, but the merge boundary must handle both.
        target.touch()
    monkeypatch.setattr(etf, "DATA", tmp_path)
    monkeypatch.setattr(etf, "parquet_path", lambda *args: target)
    monkeypatch.setattr(etf, "load_timeframe_cfg", lambda *args: ("regular", 40))
    monkeypatch.setattr(etf, "load_etf_symbols", lambda: ["SPY"])
    monkeypatch.setattr(etf, "load_eod", lambda *args, **kw: incoming)
    original_load = storage.load_parquet
    monkeypatch.setattr(storage, "load_parquet", lambda path: existing if path == target else original_load(path))
    monkeypatch.setattr(etf, "initialize_indicator_engine", lambda *args: None)
    expected = update_fixed_window(bars(), bars().iloc[:-1] if has_existing else pd.DataFrame(), 40)
    def indicators(value, **kwargs):
        assert kwargs["timeframe"] == timeframe
        pd.testing.assert_frame_equal(value, expected)
        assert value.attrs["units"] == "USD"
        assert value.attrs.get("source_note") == ("durable" if has_existing else None)
        assert "eod_provenance" not in value.attrs
        value["atr_test"] = _atr(value, 5)
        return value
    monkeypatch.setattr(etf, "apply_core", indicators)
    etf.ingest_etf_timeframe(timeframe)
    saved = pd.read_parquet(target)
    pd.testing.assert_frame_equal(saved[expected.columns], expected, check_freq=False)
    assert saved.attrs == ({"source_note": "durable", "units": "USD"} if has_existing else {"units": "USD"})
    assert saved.atr_test.iloc[-1] == 4.
    snapshot = pd.read_parquet(tmp_path / f"snapshot_etf_{timeframe}.parquet")
    assert list(snapshot.symbol) == ["SPY"]
    assert "eod_provenance" in incoming.attrs and "eod_provenance" in existing.attrs
    assert "ambiguous" not in capsys.readouterr().out


@pytest.mark.parametrize("empty_universe", [False, True])
def test_required_empty_etf_fails_instead_of_success(monkeypatch, tmp_path, empty_universe):
    monkeypatch.setattr(etf, "DATA", tmp_path)
    monkeypatch.setattr(etf, "load_timeframe_cfg", lambda *args: ("regular", 40))
    monkeypatch.setattr(etf, "initialize_indicator_engine", lambda *args: None)
    monkeypatch.setattr(etf, "load_etf_symbols", lambda: [] if empty_universe else ["SPY"])
    monkeypatch.setattr(etf, "load_eod", lambda *args, **kw: pd.DataFrame())
    monkeypatch.setattr(storage, "save_parquet", lambda *args: pytest.fail("unexpected write"))
    with pytest.raises(RuntimeError, match="Required ETF"):
        etf.ingest_etf_timeframe("weekly")


@pytest.mark.parametrize("timeframe", ["daily", "weekly"])
@pytest.mark.parametrize("empty_at", ["acquisition", "merge"])
def test_isolated_empty_etf_does_not_abort_or_reuse_stale_snapshot(
    monkeypatch, tmp_path, timeframe, empty_at, capsys,
):
    monkeypatch.setattr(etf, "DATA", tmp_path)
    monkeypatch.setattr(etf, "load_timeframe_cfg", lambda *args: ("regular", 40))
    monkeypatch.setattr(etf, "initialize_indicator_engine", lambda *args: None)
    monkeypatch.setattr(etf, "load_etf_symbols", lambda: ["MISSING", "GOOD"])
    stale = bars().iloc[-1:].copy();stale["symbol"] = "MISSING"
    snapshot_path = tmp_path / f"snapshot_etf_{timeframe}.parquet"
    storage.save_parquet(stale, snapshot_path)
    calls = []
    def load(symbol, **kwargs):
        calls.append(symbol)
        if symbol == "MISSING" and empty_at == "acquisition":return pd.DataFrame()
        result = bars();result.attrs["eod_provenance"] = provenance()
        result.attrs["test_symbol"] = symbol
        return result
    monkeypatch.setattr(etf, "load_eod", load)
    if empty_at == "merge":
        monkeypatch.setattr(etf, "update_fixed_window", lambda new, old, window:
                            pd.DataFrame() if new.attrs["test_symbol"] == "MISSING"
                            else update_fixed_window(new, old, window))
    monkeypatch.setattr(etf, "apply_core", lambda value, **kwargs: value)
    etf.ingest_etf_timeframe(timeframe)
    assert calls == ["MISSING", "GOOD"]
    snapshot = pd.read_parquet(snapshot_path)
    assert list(snapshot.symbol) == ["GOOD"]
    assert not (tmp_path / f"bars/stocks_{timeframe}/MISSING.parquet").exists()
    assert "eod_provenance" not in snapshot.attrs
    assert f"excluding {timeframe}/MISSING" in capsys.readouterr().out


@pytest.mark.parametrize("weekly_count,expected_count,passes", [(25, 2335, False), (89, 100, False), (90, 100, True), (100, 100, True)])
def test_family_gate_preserves_previous_generation_and_mirrors(monkeypatch, tmp_path, weekly_count, expected_count, passes):
    symbols = [f"S{i}" for i in range(expected_count)]
    old = "prod-20261016T231814Z-1b0d93c8";new = "prod-20261020T000826Z-370326cf"
    snapshot = bars().iloc[-1:].copy();snapshot["symbol"] = symbols[0];snapshot.attrs = {}
    publish_stock_eod_family(old, snapshot, snapshot, snapshot, expected_pointer_revision=None,
                             expected_previous_family_run_id=None, data_root=tmp_path)
    pointer = current_pointer_path(tmp_path);previous_bytes = pointer.read_bytes()
    for tf in family.TIMEFRAMES:storage.save_parquet(snapshot, tmp_path / f"snapshot_stocks_{tf}.parquet")
    old_mirrors = {tf: (tmp_path / f"snapshot_stocks_{tf}.parquet").read_bytes() for tf in family.TIMEFRAMES}
    monkeypatch.setattr(family, "DATA", tmp_path)
    monkeypatch.setattr(run_timeframe, "DATA", tmp_path)
    monkeypatch.setattr(family, "resolve_family_symbols", lambda: symbols)
    monkeypatch.setattr(family, "initialize_indicator_engine", lambda *args: None)
    monkeypatch.setattr(family, "acquire_reference_data", lambda **kw: ({}, pd.DataFrame(), 6, 6, 0))
    monkeypatch.setattr(family, "family_manifest", lambda value: pd.DataFrame({"interval": ["1d", "1wk", "1mo"]}))
    observation = SimpleNamespace(data_quality_status="valid", finalized_frame=bars())
    monkeypatch.setattr(family, "reconcile_stock_eod_family", lambda value: SimpleNamespace(daily=observation, weekly=observation, monthly=observation))
    monkeypatch.setattr(family, "process_one_preloaded", lambda *args, **kw: SimpleNamespace(accepted=True))
    monkeypatch.setattr(family, "build_timeframe_snapshot_frame", lambda ns, tf, *args: pd.DataFrame({"symbol": symbols[:weekly_count] if tf == "weekly" else symbols}))
    acquisition = lambda *args, **kw: SimpleNamespace(requests=SimpleNamespace(attempted=3, succeeded=3, failed=0, reconciliation_provider_requests=0))
    report = tmp_path / "coverage.json"
    if passes:
        family.run_family(run_id=new, acquire=acquisition, coverage_report=report)
        assert json.loads(pointer.read_text())["family_run_id"] == new
    else:
        monkeypatch.setattr(family, "publish_stock_eod_family", lambda *args, **kw: pytest.fail("publication reached"))
        with pytest.raises(RuntimeError, match="catastrophic coverage"):
            family.run_family(run_id=new, acquire=acquisition, coverage_report=report)
        assert pointer.read_bytes() == previous_bytes
        assert not (tmp_path / "snapshots/stocks/eod/generations" / new).exists()
        assert all((tmp_path / f"snapshot_stocks_{tf}.parquet").read_bytes() == old_mirrors[tf] for tf in family.TIMEFRAMES)
    assert json.loads(report.read_text())["passed"] == passes
    assert family.production_manifest_path(new).exists()  # telemetry survives


@pytest.mark.parametrize("failed_stage", ["family", "weekly", "daily"])
@pytest.mark.parametrize("manual", [False, True])
def test_profile_failure_never_marks_signal_cycle_success(monkeypatch, tmp_path, failed_stage, manual):
    spec = importlib.util.spec_from_file_location("hotfix_profile", Path(__file__).resolve().parents[1] / ".github/scripts/run_stocks_eod_guarded.py")
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    calls = []; notices = []
    report = {"passed": True}
    def command(cmd, check):
        calls.append(cmd)
        stage = "family" if "--coverage-report" in cmd else cmd[-1]
        if stage == "family":Path(cmd[-1]).write_text(json.dumps(report))
        if stage == failed_stage:raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(runner.subprocess, "run", command)
    monkeypatch.setattr(runner, "notify_stock_eod_failure", lambda *args: notices.append(args))
    monkeypatch.setattr(runner, "notify_combo_signals", lambda *args, **kw: pytest.fail("success notification"))
    monkeypatch.setattr(guard, "should_run_from_registry", lambda **kw: (True, "due"))
    monkeypatch.setattr(guard, "mark_registry_execution", lambda **kw: pytest.fail("registry success"))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch" if manual else "schedule")
    with pytest.raises(subprocess.CalledProcessError):runner.main()
    assert len(calls) == {"family": 1, "weekly": 2, "daily": 3}[failed_stage]
    assert len(notices) == 1
