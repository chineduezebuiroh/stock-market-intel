from __future__ import annotations

import pandas as pd
import pytest

from core import storage
from core.stock_eod_family import publish_stock_eod_family
from jobs import run_combo
from jobs import run_timeframe

RUN_ID = "prod-20260926T100000Z-0000000b"


def snapshot(symbol="AAPL", close=1.0):
    return pd.DataFrame({"close": [close], "symbol": [symbol]}, index=pd.to_datetime(["2026-09-26"]))


@pytest.fixture(autouse=True)
def local(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_DATA_BACKEND", "local")
    monkeypatch.setattr(run_combo, "DATA", tmp_path)
    publish_stock_eod_family(
        RUN_ID, snapshot(close=1), snapshot(close=2), snapshot(close=3),
        expected_pointer_revision=None, expected_previous_family_run_id=None, data_root=tmp_path,
    )
    for timeframe in ("daily", "weekly", "monthly"):
        snapshot(symbol="WRONG", close=99).to_parquet(tmp_path / f"snapshot_stocks_{timeframe}.parquet")
    return tmp_path


def configs():
    return {"stocks": {
        "one": {"lower_tf": "daily", "middle_tf": "weekly", "upper_tf": "monthly"},
        "two": {"lower_tf": "daily", "middle_tf": "weekly", "upper_tf": "monthly"},
        "other": {"lower_tf": "weekly", "middle_tf": "monthly", "upper_tf": "quarterly"},
    }, "futures": {"future": {
        "lower_tf": "daily", "middle_tf": "weekly", "upper_tf": "monthly",
    }}}


@pytest.mark.parametrize("name", ["one", "two"])
def test_dwm_configs_resolve_once_ignore_legacy_and_attach_lineage(monkeypatch, name):
    calls = 0
    real = run_combo.resolve_current_stock_eod_family

    def resolve(**kwargs):
        nonlocal calls
        calls += 1
        return real(**kwargs)

    monkeypatch.setattr(run_combo, "resolve_current_stock_eod_family", resolve)
    result = run_combo.build_combo_df("stocks", name, configs())
    assert calls == 1
    assert result.symbol.tolist() == ["AAPL"]
    assert result.lower_close.tolist() == [1]
    assert result.middle_close.tolist() == [2]
    assert result.upper_close.tolist() == [3]
    assert result.stock_eod_family_run_id.tolist() == [RUN_ID]
    assert result.stock_eod_family_committed_at_utc.notna().all()


def test_non_dwm_and_futures_keep_legacy_loader(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_combo, "_load_role_frame",
        lambda namespace, timeframe, role: calls.append((namespace, timeframe, role))
        or pd.DataFrame({f"{role}_close": [1]}, index=pd.Index(["AAPL"], name="symbol")),
    )
    monkeypatch.setattr(
        run_combo, "resolve_current_stock_eod_family",
        lambda **kwargs: pytest.fail("family resolver used"),
    )
    assert not run_combo.build_combo_df("stocks", "other", configs()).empty
    assert not run_combo.build_combo_df("futures", "future", configs()).empty
    assert len(calls) == 6


def test_lineage_survives_current_and_history_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(run_combo, "MTF_CFG", configs())
    monkeypatch.setattr(run_combo, "attach_etf_trends_for_options_combo", lambda df, **kwargs: df)
    monkeypatch.setattr(run_combo, "basic_signal_logic", lambda *args: args[-1])
    run_combo.run("stocks", "one")
    current = pd.read_parquet(tmp_path / "combo_one.parquet")
    history = list((tmp_path / "combo_history/stocks/one").glob("*.parquet"))
    assert current.stock_eod_family_run_id.tolist() == [RUN_ID]
    assert pd.read_parquet(history[0]).stock_eod_family_run_id.tolist() == [RUN_ID]


def test_real_snapshot_construction_publication_resolution_and_combo_handoff(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(run_timeframe, "DATA", tmp_path)
    monkeypatch.setattr(run_timeframe, "get_snapshot_base_cols", lambda *args: ["close"])
    for timeframe, close in (("daily", 11), ("weekly", 22), ("monthly", 33)):
        path = tmp_path / f"bars/stocks_{timeframe}/AAPL.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"open": [close], "high": [close], "low": [close], "close": [close],
             "adj_close": [close], "volume": [100]},
            index=pd.to_datetime(["2026-09-26"]),
        ).to_parquet(path)
    frames = {
        timeframe: run_timeframe.build_timeframe_snapshot_frame(
            "stocks", timeframe, ["AAPL"], set(),
        )
        for timeframe in ("daily", "weekly", "monthly")
    }
    root = tmp_path / "integrated"
    monkeypatch.setattr(run_combo, "DATA", root)
    publish_stock_eod_family(
        "prod-20260926T120000Z-0000000e", frames["daily"], frames["weekly"], frames["monthly"],
        expected_pointer_revision=None, expected_previous_family_run_id=None, data_root=root,
    )
    combo = run_combo.build_combo_df("stocks", "one", configs())
    assert combo.loc[0, ["lower_close", "middle_close", "upper_close"]].tolist() == [11, 22, 33]
