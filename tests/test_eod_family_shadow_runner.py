from pathlib import Path

import pandas as pd

from jobs import run_stock_eod_family_shadow as runner


def test_shadow_runner_writes_only_shadow_artifacts(monkeypatch, tmp_path):
    writes = []

    def loader(symbol, timeframe, **kwargs):
        interval = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[timeframe]
        result = pd.DataFrame(
            {"open": [99.0], "high": [101.0], "low": [98.0], "close": [100.0],
             "adj_close": [99.5], "volume": [1000.0]},
            index=[pd.Timestamp("2026-09-23")],
        )
        metadata = {"priceHint": 2}
        if interval != "1d":
            metadata["lastTrade"] = {"Price": 100.0, "Time": pd.Timestamp("2026-09-23")}
        result.attrs["eod_provenance"] = {
            "source_interval": interval, "adjustment_mode": "raw_unadjusted",
            "session_semantics": "regular", "provider_metadata": metadata,
        }
        return result

    real_acquire = runner.acquire_stock_eod_family
    monkeypatch.setattr(runner, "acquire_stock_eod_family",
                        lambda run_id, symbol: real_acquire(run_id, symbol, loader=loader))
    monkeypatch.setattr(runner, "initialize_indicator_engine", lambda: None)
    monkeypatch.setattr(runner, "apply_core", lambda frame, **kwargs: frame)
    monkeypatch.setattr(
        runner, "write_shadow_manifest",
        lambda frame, run_id: writes.append((frame.copy(), tmp_path / "_shadow/eod_family" / run_id / "manifest.parquet"))
        or writes[-1][1],
    )
    monkeypatch.setattr(runner.storage, "save_parquet",
                        lambda frame, path: writes.append((frame.copy(), Path(path))))

    manifest_path, indicator_path = runner.run_shadow(["AAPL"], "safe-run")

    assert manifest_path == tmp_path / "_shadow/eod_family/safe-run/manifest.parquet"
    assert indicator_path.name == "indicator_terminals.parquet"
    assert len(writes[0][0].query("record_type == 'interval'")) == 3
    summary = writes[0][0].query("record_type == 'run_summary'").iloc[0]
    assert summary.requests_attempted == 3
    assert summary.reconciliation_provider_requests == 0
    for _, path in writes:
        assert "_shadow/eod_family/safe-run" in str(path)
        assert "snapshot_stocks" not in str(path)
        assert "combo_" not in str(path)
        assert "execution_registry" not in str(path)


def test_symbol_modes_are_deduplicated(monkeypatch):
    args = type("Args", (), {
        "symbols": ["aapl", "MSFT"], "live_pattern": True,
        "universe": "shortlist_stocks",
    })()
    monkeypatch.setattr(runner, "symbols_for_universe", lambda _: ["AAPL", "TSLA"])
    symbols = runner._symbols(args)
    assert symbols == sorted(set(runner.LIVE_PATTERN_SYMBOLS) | {"AAPL", "MSFT", "TSLA"})
