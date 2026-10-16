"""Deterministic timing and unchanged processing/storage contract checks."""
import json

import pandas as pd
import pytest

from diagnostics import stock_eod_performance as perf
from jobs import run_timeframe as processing


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def summaries(capsys):
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("[STOCK_EOD_PERF] ")]
    assert len(lines) == 1
    assert len(lines[0]) < 4096
    return json.loads(lines[0].split(" ", 1)[1])


@pytest.fixture
def harness(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(perf.time, "perf_counter", clock)
    monkeypatch.setattr(processing, "DATA", tmp_path)
    incoming = pd.DataFrame(
        {"open": [99.0], "high": [101.0], "low": [98.0], "close": [100.0],
         "adj_close": [100.0], "volume": [1000.0]},
        index=pd.DatetimeIndex(["2026-10-15"], name="date"),
    )
    events, saved = [], []

    def operation(name, seconds):
        events.append(name)
        clock.advance(seconds)

    def exists(path):
        operation("exists", 1)
        return True

    def read(path):
        operation("read", 2)
        return incoming.copy(deep=True)

    def write(value, path):
        operation("write", 3)
        saved.append(value.copy(deep=True))

    def indicators(value, **kwargs):
        operation("indicator", 5)
        result = value.copy(deep=True)
        result["test_indicator"] = 42.0
        return result

    merge = processing.merge_stock_eod_window

    def measured_merge(*args, **kwargs):
        operation("merge", 7)
        return merge(*args, **kwargs)

    monkeypatch.setattr(processing.storage, "exists", exists)
    monkeypatch.setattr(processing.storage, "load_parquet", read)
    monkeypatch.setattr(processing.storage, "save_parquet", write)
    monkeypatch.setattr(processing, "apply_core", indicators)
    monkeypatch.setattr(processing, "merge_stock_eod_window", measured_merge)
    monkeypatch.setattr(processing, "safe_load_eod", lambda *a, **k: pytest.fail("provider called"))
    monkeypatch.setattr(processing, "get_snapshot_base_cols", lambda *a: list(incoming.columns))
    return clock, incoming, events, saved


def invoke(incoming, timeframe="daily", progress_index=1):
    return processing.process_one_preloaded(
        "stocks", timeframe, "AAPL", incoming, 260, progress_index=progress_index,
    )


def test_processing_bins_order_output_and_timeframe_separation(harness, capsys):
    clock, incoming, events, saved = harness
    baseline = invoke(incoming)
    baseline_events, baseline_frame = events[:], saved[-1].copy()
    assert baseline_events == ["exists", "read", "merge", "indicator", "write", "exists", "read"]
    events.clear()

    @perf.family_performance
    def run():
        perf.identify_family("test-family")
        for tf, count in zip(perf.TIMEFRAMES, (1, 2, 3)):
            for _ in range(count):
                assert invoke(incoming, tf) == baseline
                pd.testing.assert_frame_equal(saved[-1], baseline_frame)
        return "original result"

    assert run() == "original result"
    result = summaries(capsys)
    assert result["family_run_id"] == "test-family"
    assert result["timing_errors"] == 0
    assert events == baseline_events * 6
    for tf, count in zip(perf.TIMEFRAMES, (1, 2, 3)):
        bins = result["processing"][tf]
        assert bins["count"] == count
        assert bins["total_seconds"] == 21 * count
        assert bins["storage_seconds"] == 9 * count
        assert bins["indicator_seconds"] == 5 * count
        assert bins["residual_seconds"] == 7 * count
        assert sum(bins[key] for key in (
            "storage_seconds", "indicator_seconds", "residual_seconds",
        )) == bins["total_seconds"]
        assert bins["storage"] == {
            "exists": {"count": 2 * count, "total_seconds": 2 * count},
            "read": {"count": 2 * count, "total_seconds": 4 * count},
            "write": {"count": count, "total_seconds": 3 * count},
        }
    assert perf._ACTIVE.get() is None


def test_snapshot_scans_have_separate_timings_and_unchanged_contents(harness, capsys):
    _, incoming, events, _ = harness
    baseline = processing.build_timeframe_snapshot_frame("stocks", "daily", ["AAPL", "MSFT"], {"MSFT"})
    assert events == ["exists", "read"]
    events.clear()

    @perf.family_performance
    def run():
        for tf in perf.TIMEFRAMES:
            snapshot = processing.build_timeframe_snapshot_frame("stocks", tf, ["AAPL", "MSFT"], {"MSFT"})
            pd.testing.assert_frame_equal(snapshot, baseline)

    run()
    result = summaries(capsys)
    assert events == ["exists", "read"] * 3
    for tf in perf.TIMEFRAMES:
        assert result["snapshot"][tf] == {"count": 1, "total_seconds": 3}
        assert result["processing"][tf]["count"] == 0
        assert result["processing"][tf]["storage_seconds"] == 0


@pytest.mark.parametrize("failure", ["exists", "read", "indicator", "write"])
def test_original_exception_identity_order_and_partial_timing(harness, monkeypatch, capsys, failure):
    clock, incoming, events, _ = harness
    original = RuntimeError("original " + failure)
    target, name = (processing, "apply_core") if failure == "indicator" else (
        processing.storage, {"exists": "exists", "read": "load_parquet", "write": "save_parquet"}[failure],
    )

    def fail(*args, **kwargs):
        events.append(failure)
        clock.advance(11)
        raise original

    monkeypatch.setattr(target, name, fail)

    @perf.family_performance
    def run():
        invoke(incoming)

    for call in (lambda: invoke(incoming), run):
        events.clear()
        with pytest.raises(RuntimeError) as caught:
            call()
        assert caught.value is original
        expected = ["exists", "read", "merge", "indicator", "write"]
        assert events == expected[:expected.index(failure) + 1]
    result = summaries(capsys)["processing"]["daily"]
    assert result["count"] == 1
    assert result["total_seconds"] >= 11
    assert result["residual_seconds"] == (7 if failure in {"indicator", "write"} else 0)


def test_early_rejection_counts_without_indicator_or_write(harness, capsys):
    _, incoming, events, _ = harness
    incoming.loc[:, "close"] = float("nan")

    @perf.family_performance
    def run():
        return invoke(incoming, "weekly")

    assert not run().accepted
    assert events == ["exists", "read", "merge"]
    bins = summaries(capsys)["processing"]["weekly"]
    assert bins["count"] == 1
    assert bins["total_seconds"] == 10
    assert bins["storage_seconds"] == 3
    assert bins["indicator_seconds"] == 0
    assert bins["residual_seconds"] == 7


@pytest.mark.parametrize("verification", ["exists", "read"])
def test_verification_errors_remain_caught(harness, monkeypatch, capsys, verification):
    _, incoming, events, _ = harness
    name = "exists" if verification == "exists" else "load_parquet"
    original = getattr(processing.storage, name)
    calls = 0

    def verify(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            events.append(verification)
            raise OSError("verification failure")
        return original(path)

    monkeypatch.setattr(processing.storage, name, verify)

    @perf.family_performance
    def run():
        return invoke(incoming)

    assert run().accepted
    result = summaries(capsys)["processing"]["daily"]
    assert result["storage"][verification]["count"] == 2
    assert events.count("indicator") == events.count("write") == 1


@pytest.mark.parametrize("fault", ["clock", "clock_end", "summary", "output", "setup"])
@pytest.mark.parametrize("production_failure", [False, True])
def test_instrumentation_failures_never_replace_production_outcome(
    harness, monkeypatch, fault, production_failure,
):
    _, incoming, _, saved = harness
    original = RuntimeError("production error")

    def broken(*args, **kwargs):
        raise OSError("diagnostic failure")

    if fault == "clock":
        monkeypatch.setattr(perf.time, "perf_counter", broken)
    elif fault == "clock_end":
        calls = 0

        def end_failure():
            nonlocal calls
            calls += 1
            if calls % 2 == 0:
                raise OSError("diagnostic clock failure")
            return float(calls)

        monkeypatch.setattr(perf.time, "perf_counter", end_failure)
    elif fault == "summary":
        monkeypatch.setattr(perf.FamilyPerformance, "summary", broken)
    elif fault == "setup":
        monkeypatch.setattr(perf, "FamilyPerformance", broken)
    else:
        monkeypatch.setattr(perf, "print", broken, raising=False)

    @perf.family_performance
    def run():
        result = invoke(incoming)
        if production_failure:
            raise original
        return result

    if production_failure:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is original
    else:
        assert run().accepted
    assert len(saved) == 1
    assert perf._ACTIVE.get() is None


def test_no_collection_or_clock_outside_family(harness, monkeypatch, capsys):
    _, incoming, _, _ = harness
    monkeypatch.setattr(perf.time, "perf_counter", lambda: pytest.fail("inactive timer"))
    assert invoke(incoming).accepted
    assert "[STOCK_EOD_PERF]" not in capsys.readouterr().out


def test_clock_failure_marks_summary_incomplete(harness, monkeypatch, capsys):
    _, incoming, _, _ = harness

    def broken():
        raise OSError("timer unavailable")

    monkeypatch.setattr(perf.time, "perf_counter", broken)

    @perf.family_performance
    def run():
        return invoke(incoming)

    assert run().accepted
    result = summaries(capsys)
    assert result["timing_errors"] > 0
    assert result["processing"]["daily"]["count"] == 1


def test_scope_resets_between_families(harness, capsys):
    _, incoming, _, _ = harness

    @perf.family_performance
    def run():
        invoke(incoming)

    for _ in range(2):
        run()
        assert summaries(capsys)["processing"]["daily"]["count"] == 1


def test_new_symbol_storage_order_and_counts(harness, monkeypatch, capsys):
    clock, incoming, events, _ = harness
    calls = 0

    def exists(path):
        nonlocal calls
        calls += 1
        events.append("exists")
        clock.advance(1)
        return calls != 1

    monkeypatch.setattr(processing.storage, "exists", exists)

    @perf.family_performance
    def run():
        return invoke(incoming, progress_index=2)

    assert run().accepted
    assert events == ["exists", "merge", "indicator", "write", "exists"]
    bins = summaries(capsys)["processing"]["daily"]
    assert bins["storage"]["read"]["count"] == 0
    assert bins["total_seconds"] == 17
    assert bins["storage_seconds"] == 5
    assert bins["residual_seconds"] == 7


def test_snapshot_exception_is_preserved_and_timed(harness, monkeypatch, capsys):
    clock, _, _, _ = harness
    original = OSError("snapshot read failed")

    def read(path):
        clock.advance(5)
        raise original

    monkeypatch.setattr(processing.storage, "load_parquet", read)

    @perf.family_performance
    def run():
        processing.build_timeframe_snapshot_frame("stocks", "monthly", ["AAPL"])

    with pytest.raises(OSError) as caught:
        run()
    assert caught.value is original
    result = summaries(capsys)
    assert result["snapshot"]["monthly"] == {"count": 1, "total_seconds": 6}


def test_real_publication_cas_failure_and_pointer_unchanged(monkeypatch, tmp_path, capsys):
    from core import storage
    from core import stock_eod_family as publication

    monkeypatch.setattr(storage, "_DATA_BACKEND", "local")
    snapshot = pd.DataFrame({"symbol": ["AAPL"], "close": [100.0]})

    @perf.family_performance
    def publish(run_id):
        return publication.publish_stock_eod_family(
            run_id, snapshot, snapshot, snapshot,
            expected_pointer_revision=None, expected_previous_family_run_id=None,
            data_root=tmp_path,
        )

    committed = publish("prod-20261016T000000Z-00000001")
    assert summaries(capsys)["timing_errors"] == 0
    resolved = publication.resolve_current_stock_eod_family(data_root=tmp_path)
    pd.testing.assert_frame_equal(resolved.daily, snapshot)
    pd.testing.assert_frame_equal(resolved.weekly, snapshot)
    pd.testing.assert_frame_equal(resolved.monthly, snapshot)
    with pytest.raises(publication.ConcurrentStockEODPublication):
        publish("prod-20261016T010000Z-00000002")
    summaries(capsys)
    assert publication.resolve_current_stock_eod_family(data_root=tmp_path).pointer == committed.pointer
