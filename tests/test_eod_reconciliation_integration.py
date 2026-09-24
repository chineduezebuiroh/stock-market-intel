import numpy as np
import pandas as pd

from jobs import run_timeframe


def native_frame(close, interval, *, last_trade=False):
    result = pd.DataFrame(
        {"open": [99.0], "high": [101.0], "low": [98.0], "close": [close],
         "adj_close": [np.nan], "volume": [1000.0]},
        index=[pd.Timestamp("2026-09-23")],
    )
    metadata = {"priceHint": 2}
    if last_trade:
        metadata["lastTrade"] = {"Price": 100.0, "Time": pd.Timestamp("2026-09-23")}
    result.attrs["eod_provenance"] = {
        "source_interval": interval, "adjustment_mode": "raw_unadjusted",
        "session_semantics": "regular", "provider_metadata": metadata,
    }
    return result


def test_historical_crwd_persisted_shape_without_session_evidence_fails_closed(monkeypatch):
    daily = native_frame(np.nan, "1d")
    weekly = native_frame(np.nan, "1wk")
    monthly = native_frame(100.0, "1mo")
    monkeypatch.setattr(
        run_timeframe, "safe_load_eod",
        lambda symbol, timeframe, **kwargs: weekly if timeframe == "weekly" else monthly,
    )
    repaired, provenance = run_timeframe._reconcile_malformed_daily_close(
        "CRWD", daily, session="regular"
    )
    assert pd.isna(repaired.iloc[-1].close)
    assert provenance["data_quality_status"] == "excluded"
    assert provenance["terminal_close_pattern"] == "1d:malformed/1wk:malformed/1mo:valid"


def test_historical_shape_repairs_only_with_bound_single_monthly_provenance(monkeypatch):
    daily = native_frame(np.nan, "1d")
    weekly = native_frame(np.nan, "1wk")
    monthly = native_frame(100.0, "1mo", last_trade=True)
    monkeypatch.setattr(
        run_timeframe, "safe_load_eod",
        lambda symbol, timeframe, **kwargs: weekly if timeframe == "weekly" else monthly,
    )
    repaired, provenance = run_timeframe._reconcile_malformed_daily_close(
        "CRWD", daily, session="regular"
    )
    assert repaired.iloc[-1].close == 100
    assert pd.isna(repaired.iloc[-1].adj_close)
    assert provenance["evidence_tier"] == "tier_2_provenance_bound_single_source"


def test_evidence_fetch_exception_is_symbol_local(monkeypatch):
    daily = native_frame(np.nan, "1d")
    monthly = native_frame(100.0, "1mo", last_trade=True)

    def fetch(symbol, timeframe, **kwargs):
        if timeframe == "weekly":
            raise RuntimeError("provider failure")
        return monthly

    monkeypatch.setattr(run_timeframe, "safe_load_eod", fetch)
    repaired, provenance = run_timeframe._reconcile_malformed_daily_close(
        "CRWD", daily, session="regular"
    )
    assert repaired.iloc[-1].close == 100
    assert provenance["terminal_close_pattern"] == "1d:malformed/1wk:unavailable/1mo:valid"


def test_ingest_repairs_before_indicators_and_evidence_fetches_have_no_wm_side_effects(
    monkeypatch, tmp_path
):
    daily = native_frame(np.nan, "1d")
    weekly = native_frame(100.0, "1wk", last_trade=True)
    monthly = native_frame(100.0, "1mo", last_trade=True)
    calls, saved, indicator_inputs = [], [], []

    def fetch(symbol, timeframe, **kwargs):
        calls.append(timeframe)
        return {"daily": daily, "weekly": weekly, "monthly": monthly}[timeframe]

    monkeypatch.setattr(run_timeframe, "safe_load_eod", fetch)
    monkeypatch.setattr(run_timeframe, "parquet_path", lambda *args: tmp_path / "daily.parquet")
    monkeypatch.setattr(run_timeframe.storage, "exists", lambda path: False)
    monkeypatch.setattr(run_timeframe.storage, "save_parquet", lambda df, path: saved.append(str(path)))
    monkeypatch.setattr(
        run_timeframe, "apply_core",
        lambda df, **kwargs: indicator_inputs.append(df.copy()) or df,
    )
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_ATTEMPTS", 0)
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_ELIGIBLE", 0)
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_SKIPPED", 0)
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_TELEMETRY", [])

    rejected = run_timeframe.ingest_one("stocks", "daily", ["CRWD"], "regular", 260)

    assert not rejected
    assert calls == ["daily", "weekly", "monthly"]
    assert indicator_inputs[0].iloc[-1].close == 100
    assert pd.isna(indicator_inputs[0].iloc[-1].adj_close)
    assert not any("weekly" in path or "monthly" in path for path in saved)


def test_health_write_failure_is_visible_but_nonfatal(monkeypatch, capsys):
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_TELEMETRY", [{"status": "test"}])
    monkeypatch.setattr(
        run_timeframe.storage, "save_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("health unavailable")),
    )
    run_timeframe.write_eod_reconciliation_telemetry("daily")
    assert "[HEALTH][WARN]" in capsys.readouterr().out


def test_budget_selection_is_neutral_rotating_and_exhaustion_is_measurable(
    monkeypatch, tmp_path
):
    symbols = ["ZZZ", "AAA", "MMM"]
    daily = native_frame(np.nan, "1d")
    weekly = native_frame(100.0, "1wk", last_trade=True)
    monthly = native_frame(100.0, "1mo", last_trade=True)
    fetched = []
    health_frames = []

    def fetch(symbol, timeframe, **kwargs):
        fetched.append((symbol, timeframe))
        return {"daily": daily, "weekly": weekly, "monthly": monthly}[timeframe]

    monkeypatch.setattr(run_timeframe, "safe_load_eod", fetch)
    monkeypatch.setattr(run_timeframe, "parquet_path", lambda *args: tmp_path / f"{args[-1]}.parquet")
    monkeypatch.setattr(run_timeframe.storage, "exists", lambda path: False)
    monkeypatch.setattr(run_timeframe.storage, "save_parquet", lambda df, path: health_frames.append(df.copy()))
    monkeypatch.setattr(run_timeframe, "apply_core", lambda df, **kwargs: df)
    monkeypatch.setattr(run_timeframe, "MAX_EOD_RECONCILIATION_SYMBOLS", 1)
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_ATTEMPTS", 0)
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_ELIGIBLE", 0)
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_SKIPPED", 0)
    monkeypatch.setattr(run_timeframe, "EOD_RECONCILIATION_TELEMETRY", [])

    rejected = run_timeframe.ingest_one("stocks", "daily", symbols, "regular", 260)

    supplemental = [(symbol, timeframe) for symbol, timeframe in fetched if timeframe != "daily"]
    assert len(supplemental) == 2
    assert len(rejected) == 2
    telemetry = health_frames[-1]
    summary = telemetry.loc[telemetry["record_type"] == "run_summary"].iloc[0]
    assert summary.eligible_count == 3
    assert summary.attempted_count == 1
    assert summary.skipped_budget_count == 2
    skipped = telemetry.loc[telemetry["reason"] == "supplemental fetch budget exhausted"]
    assert set(skipped.symbol) == rejected
    assert skipped.selection_priority_sha256.notna().all()
