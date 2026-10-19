"""Exact historical-output oracle frozen before the raw callback change.

The fixture is the unmodified production module at 2d87d0d (also unchanged
on main 2824b059). It deliberately keeps the original Series callback sites.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import socket

import numpy as np
import pandas as pd
import pytest

from indicators import core
from indicators import composite_using_percentiles as percentile
from indicators.composite_spy_qqq_volume_ma_ratio import preloaded_eod_references


def original_rank(x):
    # Independent original production formula; do not delegate to a new helper.
    return len(x[x <= x.iloc[-1]]) / len(x) * 100


ORIGINAL = {"__name__": "indicators._original_percentiles", "__package__": "indicators"}
SOURCE = Path(__file__).with_name("fixtures") / "percentile_original.py.txt"
exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), ORIGINAL)
ORIGINAL["_pctrank"] = original_rank


def bars(n, *, seed=73):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, .5, n))
    open_ = close + rng.normal(0, .3, n)
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + 1,
        "low": np.minimum(open_, close) - 1, "close": close,
        "adj_close": close, "volume": rng.integers(0, 10000, n),
    }, index=pd.bdate_range(end="2026-09-23", periods=n, name="session"))


@contextmanager
def original_indicators(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setitem(core.INDICATOR_FUNCS, "exh_abs_price_action",
                      ORIGINAL["indicator_exh_abs_price_action"])
        patch.setitem(core.INDICATOR_FUNCS, "significant_volume",
                      ORIGINAL["indicator_significant_volume"])
        yield


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access forbidden in percentile equivalence tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    core.initialize_indicator_engine(Path(__file__).resolve().parents[1] / "config")
    references = {(s, tf): bars(300) for s in ("SPY", "QQQ")
                  for tf in ("daily", "weekly", "monthly")}
    with preloaded_eod_references(references):
        yield


def compare_core(monkeypatch, frame, namespace, timeframe, capsys):
    before = frame.copy(deep=True)
    capsys.readouterr()
    with original_indicators(monkeypatch):
        expected = core.apply_core(frame, namespace, timeframe)
    expected_log = capsys.readouterr().out
    actual = core.apply_core(frame, namespace, timeframe)
    actual_log = capsys.readouterr().out
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert frame.attrs == before.attrs
    assert actual_log == expected_log
    return actual


@pytest.mark.parametrize("timeframe,n,L", [("daily", 260, 126), ("weekly", 210, 104),
                                           ("monthly", 250, 60)])
@pytest.mark.parametrize("case", ["full", "empty", "short", "boundary", "nan", "inf",
                                  "constant", "nullable", "existing", "unsorted", "duplicate",
                                  "missing_volume", "missing_adj_close"])
def test_complete_core_history(monkeypatch, capsys, timeframe, n, L, case):
    frame = bars(n)
    if case == "empty":
        frame = frame.iloc[:0]
    elif case == "short":
        frame = frame.iloc[:3]
    elif case == "boundary":
        frame = frame.iloc[:L + 2]
    elif case == "nan":
        frame.iloc[[0, 15, L - 1], frame.columns.get_indexer(["close", "volume"])] = np.nan
    elif case == "inf":
        frame["volume"] = frame.volume.astype(float)
        frame.iloc[[4, L], frame.columns.get_indexer(["close", "volume"])] = np.inf
    elif case == "constant":
        frame.loc[:, :] = 0
    elif case == "nullable":
        frame = frame.astype({c: "Float64" for c in frame})
        frame.iloc[5, :] = pd.NA
    elif case == "existing":
        for inst in reversed(core._STORAGE_PROFILES[("stocks", timeframe)]):
            frame[inst.key] = -999.0
    elif case == "unsorted":
        frame = frame.iloc[::-1]
    elif case == "duplicate":
        frame = pd.concat([frame, frame.iloc[-2:]])
    elif case.startswith("missing_"):
        frame = frame.drop(columns=case.removeprefix("missing_"))
    frame.attrs = {"eod_provenance": {"source": "synthetic", "repaired": False}}
    compare_core(monkeypatch, frame, "stocks", timeframe, capsys)


def test_other_configured_namespaces_and_lookbacks(monkeypatch, capsys):
    # Make the non-EOD benchmark boundary deterministic and provider-closed.
    from indicators import composite_spy_qqq_volume_ma_ratio as benchmark
    def references(tf, length):
        v = bars(300).volume.astype(float).rolling(length, min_periods=length).mean()
        return v, v
    with monkeypatch.context() as patch:
        patch.setattr(benchmark, "_spy_qqq_vol_ma_for_timeframe", references)
        for namespace, timeframe in core._STORAGE_PROFILES:
            if namespace == "stocks" and timeframe in {"daily", "weekly", "monthly"}:
                continue
            compare_core(monkeypatch, bars(260), namespace, timeframe, capsys)


@pytest.mark.parametrize("L", [1, 2, 7, 50, 60, 104, 115, 126])
def test_percentile_indicators_boundaries_and_thresholds(L):
    for n in sorted({0, 1, max(0, L - 1), L, L + 1, L + 3}):
        frame = bars(n)
        # Ties, zero range, and values adjacent to each integer rank boundary.
        frame["volume"] = np.arange(n, dtype=float) % 3
        if n:
            frame.iloc[-1, frame.columns.get_loc("volume")] = np.nextafter(1., 2.)
        for name in ("indicator_exh_abs_price_action", "indicator_significant_volume"):
            for z in (0, 1):
                kwargs = {"lookback_period": L, "z": z}
                expected = ORIGINAL[name](frame, **kwargs)
                actual = getattr(percentile, name)(frame, **kwargs)
                pd.testing.assert_series_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("L", [1, 2, 7, 50, 60, 104, 115, 126])
@pytest.mark.parametrize("case", ["random", "ties", "constant", "zeros", "nan", "inf", "nullable"])
def test_raw_callback_exact_rank_and_nan_mask(L, case):
    from indicators.helpers import _pctrank_raw
    rng = np.random.default_rng(123)
    for n in sorted({0, 1, max(0, L - 1), L, L + 1, 260}):
        x = rng.normal(size=n)
        if case == "ties":
            x = rng.integers(-3, 4, n).astype(float)
        elif case in {"constant", "zeros"}:
            x[:] = 0. if case == "zeros" else 7.
        elif case == "nan":
            x[::17] = np.nan
        elif case == "inf":
            x[::23] = np.inf
            x[::31] = -np.inf
        series = pd.Series(x, index=pd.date_range("2000-01-01", periods=n, name="date"),
                           dtype="Float64" if case == "nullable" else "float64", name="rank")
        expected = series.rolling(L).apply(original_rank, raw=False)
        actual = series.rolling(L).apply(_pctrank_raw, raw=True)
        pd.testing.assert_series_equal(actual, expected, check_exact=True)


def test_callback_failure_preserves_per_indicator_fallback(monkeypatch, capsys):
    def fail(_):
        raise RuntimeError("synthetic callback failure")
    frame = bars(260)
    with monkeypatch.context() as patch:
        patch.setitem(ORIGINAL, "_pctrank", fail)
        with original_indicators(monkeypatch):
            expected = core.apply_core(frame, "stocks", "daily")
    expected_log = capsys.readouterr().out
    with monkeypatch.context() as patch:
        patch.setattr(percentile, "_pctrank_raw", fail)
        actual = core.apply_core(frame, "stocks", "daily")
    assert capsys.readouterr().out == expected_log
    assert expected_log.count("synthetic callback failure") == 4
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("L", [-1, 0])
def test_invalid_window_preserves_original_exception_and_fallback(monkeypatch, capsys, L):
    frame = bars(260)
    for name in ("indicator_exh_abs_price_action", "indicator_significant_volume"):
        with pytest.raises(Exception) as expected:
            ORIGINAL[name](frame, lookback_period=L)
        with pytest.raises(type(expected.value)) as actual:
            getattr(percentile, name)(frame, lookback_period=L)
        assert actual.value.args == expected.value.args
    instances = core._STORAGE_PROFILES[("stocks", "daily")]
    configured = [core.IndicatorInstance(inst.key, inst.base_id,
                  {**inst.params, "lookback_period": L})
                  if inst.base_id in {"exh_abs_price_action", "significant_volume"} else inst
                  for inst in instances]
    monkeypatch.setitem(core._STORAGE_PROFILES, ("stocks", "daily"), configured)
    compare_core(monkeypatch, frame, "stocks", "daily", capsys)


def test_replacement_correction_and_window_truncation(monkeypatch, capsys):
    from etl.window import update_fixed_window
    for tf, n in (("daily", 260), ("weekly", 210), ("monthly", 250)):
        existing = bars(n + 8)
        for inst in core._STORAGE_PROFILES[("stocks", tf)]:
            existing[inst.key] = -999.
        updates = existing.iloc[[-30, -1]].copy()
        updates["close"] += .1
        updates["volume"] += 5
        updates = pd.concat([updates, bars(n + 9).iloc[-1:]])
        merged = update_fixed_window(updates, existing, n)
        assert len(merged) == n
        compare_core(monkeypatch, merged, "stocks", tf, capsys)


def test_family_persistence_provenance_coverage_and_publication(monkeypatch, tmp_path):
    from core import storage
    from core.stock_eod_family import resolve_current_stock_eod_family
    from etl.eod_family import acquire_stock_eod_family
    from jobs import run_stock_eod_family as runner, run_timeframe

    monkeypatch.setattr(storage, "_DATA_BACKEND", "local")
    monkeypatch.setenv("STOCK_EOD_CAPTURE_APPLICATION", "0")
    monkeypatch.setattr(runner, "resolve_family_symbols", lambda: ["AAPL", "MSFT"])
    fixed = datetime(2026, 9, 23, 22, tzinfo=timezone.utc)
    windows = {"daily": 260, "weekly": 210, "monthly": 250}
    def provider(symbol, timeframe, **kwargs):
        frame = bars(windows[timeframe])
        # Bound peer evidence to the same session and price for daily repair.
        frame.iloc[-1, frame.columns.get_indexer(["open", "high", "low", "adj_close"])] = [99., 101., 98., 100.]
        frame.iloc[-1, frame.columns.get_loc("close")] = 100.
        frame.attrs["eod_provenance"] = {
            "source_interval": {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe],
            "adjustment_mode": "raw_unadjusted", "session_semantics": "regular",
            "provider_metadata": {"priceHint": 2, "lastTrade": {
                "Price": 100., "Time": frame.index[-1]}},
        }
        if (symbol, timeframe) in {("AAPL", "daily"), ("MSFT", "weekly")}:
            frame.iloc[-1, frame.columns.get_loc("close")] = np.nan
        return frame
    def acquire(run_id, symbol, **kwargs):
        return acquire_stock_eod_family(run_id, symbol, loader=provider,
                                       clock=lambda: fixed, **kwargs)
    results = []
    for original in (True, False):
        root = tmp_path / ("original" if original else "optimized")
        root.mkdir()
        monkeypatch.setattr(runner, "DATA", root)
        monkeypatch.setattr(run_timeframe, "DATA", root)
        with monkeypatch.context() as patch:
            if original:
                patch.setitem(core.INDICATOR_FUNCS, "exh_abs_price_action",
                              ORIGINAL["indicator_exh_abs_price_action"])
                patch.setitem(core.INDICATOR_FUNCS, "significant_volume",
                              ORIGINAL["indicator_significant_volume"])
            path = runner.run_family(run_id="prod-20260923T220000Z-12345678",
                                     acquire=acquire,
                                     reference_loader=lambda *a, **kw: bars(300),
                                     coverage_report=root / "coverage.json")
        telemetry = storage.load_parquet(path)
        interval_rows = telemetry.loc[telemetry.record_type == "interval"]
        assert "repaired" in set(interval_rows.finalization_status)
        assert "excluded" in set(interval_rows.processing_status)
        published = resolve_current_stock_eod_family(data_root=root)
        assert set(published.weekly.symbol) == {"AAPL"}
        files = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.parquet")}
        results.append((files, json.loads((root / "coverage.json").read_text())))
    assert results[0] == results[1]
