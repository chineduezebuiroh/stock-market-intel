from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import product

import numpy as np
import pandas as pd
import pytest

from etl.eod_family import (
    FAMILY_INTERVALS,
    MANIFEST_SCHEMA,
    NativeEODObservation,
    RequestAccounting,
    StockEODObservationFamily,
    acquire_stock_eod_family,
    family_manifest,
    frame_sha256,
    reconcile_stock_eod_family,
    shadow_manifest_path,
    validity_pattern,
    write_shadow_manifest,
)
from etl.eod_reconciliation import canonical_terminal_session_close, observation_from_native_frame


UTC = timezone.utc
SESSION = pd.Timestamp("2026-09-23")


def frame(interval: str, close=100.0, *, last_trade=True, low=98.0, high=101.0):
    result = pd.DataFrame(
        {"open": [99.0], "high": [high], "low": [low], "close": [close],
         "adj_close": [99.5], "volume": [1000.0]}, index=[SESSION]
    )
    metadata = {"priceHint": 2}
    if interval != "1d" and last_trade:
        metadata["lastTrade"] = {"Price": close, "Time": SESSION}
    result.attrs["eod_provenance"] = {
        "source_interval": interval, "adjustment_mode": "raw_unadjusted",
        "session_semantics": "regular", "yfinance_version": "test",
        "provider_metadata": metadata,
    }
    return result


def observation(interval: str, close=100.0, *, failed=False, offset=0):
    started = datetime(2026, 9, 23, tzinfo=UTC) + timedelta(seconds=offset)
    raw = None if failed else frame(interval, close)
    quality = "unavailable" if failed else ("valid" if pd.notna(close) else "invalid")
    return NativeEODObservation(
        interval, raw, started, started + timedelta(seconds=1),
        "failed" if failed else "success", "boom" if failed else None,
        provider_version="test", adjustment_mode="raw_unadjusted",
        session_semantics="regular", provider_metadata={},
        payload_sha256=frame_sha256(raw), quality_status=quality,
        quality_reason="empty provider payload" if failed else None,
        terminal_market_session=None if interval != "1d" else "2026-09-23",
        raw_terminal_values={name: None for name in ("open", "high", "low", "close", "adj_close", "volume")}
        if failed else {name: float(raw.iloc[-1][name]) if pd.notna(raw.iloc[-1][name]) else None
                        for name in ("open", "high", "low", "close", "adj_close", "volume")},
    )


def family(d=100.0, w=100.0, m=100.0):
    items = [observation("1d", d, offset=0), observation("1wk", w, offset=2),
             observation("1mo", m, offset=4)]
    return StockEODObservationFamily("run-1", "CRWD", *items,
                                     RequestAccounting(3, 3, 0, FAMILY_INTERVALS))


def test_native_and_finalized_are_separate_and_hash_is_deterministic():
    source = family(d=np.nan)
    native_before = source.daily.raw_frame.copy(deep=True)
    result = reconcile_stock_eod_family(source)
    assert pd.isna(source.daily.raw_frame.iloc[-1].close)
    pd.testing.assert_frame_equal(source.daily.raw_frame, native_before)
    assert result.daily.finalized_frame is not source.daily.raw_frame
    assert result.daily.finalized_frame.iloc[-1].close == 100
    assert result.daily.finalized_frame.iloc[-1].adj_close == 99.5
    assert frame_sha256(source.daily.raw_frame) == frame_sha256(source.daily.raw_frame.copy())


def test_partial_family_and_forbidden_membership():
    partial = StockEODObservationFamily(
        "run", "AAPL", observation("1d"), None, observation("1mo"),
        RequestAccounting(3, 2, 1, FAMILY_INTERVALS),
    )
    assert [item.source_interval for item in partial.observations()] == ["1d", "1mo"]
    for interval in ("3mo", "1y", "4h"):
        with pytest.raises(ValueError, match="unsupported family interval"):
            observation("1d").__class__(interval, None, datetime.now(UTC), datetime.now(UTC), "failed")


def test_acquisition_calls_each_interval_once_isolates_failure_and_preserves_provenance():
    calls = []
    times = iter(datetime(2026, 9, 23, tzinfo=UTC) + timedelta(seconds=i) for i in range(6))

    def loader(symbol, timeframe, **kwargs):
        calls.append((symbol, timeframe, kwargs["window_bars"], kwargs["session"]))
        if timeframe == "weekly":
            raise RuntimeError("weekly unavailable")
        interval = {"daily": "1d", "monthly": "1mo"}[timeframe]
        return frame(interval)

    result = acquire_stock_eod_family("run", "aapl", loader=loader, clock=lambda: next(times))
    assert [call[1] for call in calls] == ["daily", "weekly", "monthly"]
    assert result.requests == RequestAccounting(3, 2, 1, FAMILY_INTERVALS)
    assert result.daily.provider_version == "test"
    assert result.weekly.acquisition_status == "failed"
    assert result.weekly.acquisition_error == "RuntimeError: weekly unavailable"
    assert result.monthly.fetch_started_at < result.monthly.fetch_completed_at


def test_family_reconciliation_performs_zero_provider_calls_and_matches_contract():
    source = family(d=np.nan)
    expected = canonical_terminal_session_close(
        [observation_from_native_frame(source.weekly.raw_frame),
         observation_from_native_frame(source.monthly.raw_frame)],
        target_session=SESSION.date(), price_tolerance=.005,
    )
    actual = reconcile_stock_eod_family(source)
    assert actual.reconciliation == expected
    assert source.requests.reconciliation_provider_requests == 0
    assert actual.reconciliation.evidence_tier == "tier_1_independent_consensus"


def test_tier_two_conflict_and_range_rejection_match_existing_contract():
    tier_two = family(d=np.nan, w=np.nan, m=100.0)
    assert reconcile_stock_eod_family(tier_two).reconciliation.evidence_tier == "tier_2_provenance_bound_single_source"

    conflict = reconcile_stock_eod_family(family(d=np.nan, w=100.0, m=102.0))
    assert conflict.reconciliation.status == "conflict"
    assert pd.isna(conflict.daily.finalized_frame.iloc[-1].close)

    source = family(d=np.nan)
    source.daily.raw_frame.loc[SESSION, "high"] = 99.0
    rejected = reconcile_stock_eod_family(source)
    assert rejected.daily.data_quality_status == "excluded"
    assert "high/low" in rejected.daily.repair_provenance["reason"]


def test_weekly_and_monthly_targets_are_never_repaired():
    result = reconcile_stock_eod_family(family(d=100.0, w=np.nan, m=np.nan))
    assert pd.isna(result.weekly.finalized_frame.iloc[-1].close)
    assert pd.isna(result.monthly.finalized_frame.iloc[-1].close)
    assert result.weekly.data_quality_status == "invalid"
    assert result.monthly.data_quality_status == "invalid"


@pytest.mark.parametrize("states", list(product((False, True), repeat=3)))
def test_all_eight_validity_and_close_null_patterns(states):
    values = [100.0 if valid else np.nan for valid in states]
    value, close_null = validity_pattern(family(*values))
    expected = "/".join(f"{interval}:{'valid' if valid else 'invalid'}"
                        for interval, valid in zip(FAMILY_INTERVALS, states))
    assert value == expected
    assert close_null.count("close_null=true") == states.count(False)


def test_manifest_schema_partial_errors_request_counts_and_timing():
    source = family(d=np.nan)
    # Replace W with an explicit failed observation while retaining siblings.
    source = StockEODObservationFamily(
        source.run_id, source.symbol, source.daily, observation("1wk", failed=True, offset=2),
        source.monthly, RequestAccounting(3, 2, 1, FAMILY_INTERVALS),
    )
    manifest = family_manifest(reconcile_stock_eod_family(source))
    assert tuple(manifest.columns) == MANIFEST_SCHEMA
    assert manifest.interval.tolist() == list(FAMILY_INTERVALS)
    weekly = manifest.loc[manifest.interval == "1wk"].iloc[0]
    assert weekly.acquisition_status == "failed"
    assert weekly.acquisition_error == "boom"
    assert weekly.requests_attempted == 3 and weekly.requests_failed == 1
    assert weekly.reconciliation_provider_requests == 0
    assert manifest.d_to_w_gap_seconds.notna().all()
    assert manifest.w_to_m_gap_seconds.notna().all()
    assert manifest.family_duration_seconds.notna().all()
    assert manifest.loc[manifest.interval == "1d", "reconciliation_tier"].iloc[0]


def test_shadow_storage_path_cannot_alias_authoritative_paths(tmp_path):
    path = shadow_manifest_path("run-1", data_root=tmp_path)
    assert path == tmp_path / "_shadow/eod_family/run-1/manifest.parquet"
    assert "snapshot_" not in str(path) and "combo_" not in str(path) and "_runs" not in str(path)
    with pytest.raises(ValueError):
        shadow_manifest_path("../escape", data_root=tmp_path)

    writes = []
    manifest = family_manifest(reconcile_stock_eod_family(family()))
    written = write_shadow_manifest(manifest, "run-1", data_root=tmp_path,
                                    writer=lambda df, target: writes.append((df, target)))
    assert writes[0][1] == written == path
    assert tuple(writes[0][0].columns) == MANIFEST_SCHEMA
