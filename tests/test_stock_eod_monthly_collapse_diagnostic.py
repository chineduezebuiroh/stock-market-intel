from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import storage
from diagnostics import diagnose_stock_eod_monthly_collapse as diagnostic

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/stock-eod-monthly-collapse-diagnostic.yml"


def row(symbol="AAPL", interval="1mo", **values):
    base = {
        "record_type": "interval", "symbol": symbol, "interval": interval,
        "acquisition_status": "success", "quality_status": "valid", "quality_reason": None,
        "close_is_null": False, "raw_open": 100.0, "raw_high": 102.0,
        "raw_low": 99.0, "raw_close": 101.0, "raw_adj_close": 101.0,
        "raw_volume": 1000.0, "terminal_session": "2026-10-01",
        "session_evidence": "native_daily_interval_label" if interval == "1d" else "yfinance_interval_merge_lastTrade",
        "adjustment_mode": "raw_unadjusted", "session_semantics": "regular",
        "provider_metadata_json": json.dumps({"priceHint": 2}),
        "finalization_status": "valid", "processing_status": "accepted",
        "processing_reason": None,
    }
    base.update(values)
    return base


def malformed_m(symbol="AAPL", **values):
    defaults = {
        "quality_status": "invalid", "quality_reason": "invalid terminal required fields: close",
        "close_is_null": True, "raw_close": np.nan, "raw_adj_close": np.nan,
        "terminal_session": None, "session_evidence": None,
        "finalization_status": "invalid", "processing_status": "excluded",
        "processing_reason": "finalized observation is not eligible",
    }
    defaults.update(values)
    return row(symbol, "1mo", **defaults)


def family(symbol="AAPL", d=101.0, w=101.0, **m_values):
    return [
        row(symbol, "1d", raw_close=d),
        row(symbol, "1wk", raw_close=w),
        malformed_m(symbol, **m_values),
    ]


@pytest.fixture(autouse=True)
def expected_count(monkeypatch):
    monkeypatch.setattr(diagnostic, "EXPECTED_AFFECTED", 1)


def test_exact_affected_population_selection():
    intervals = diagnostic.validate_manifest(pd.DataFrame(family() + [row("MSFT", "1mo")]))
    assert diagnostic.select_affected_monthly(intervals).symbol.tolist() == ["AAPL"]


def test_daily_and_weekly_eligibility():
    intervals = diagnostic.validate_manifest(pd.DataFrame(family()))
    symbols = {"AAPL"}
    assert diagnostic.evaluate_eligibility(intervals, symbols, "1d").loc["AAPL", "state"] == "ELIGIBLE"
    assert diagnostic.evaluate_eligibility(intervals, symbols, "1wk").loc["AAPL", "state"] == "ELIGIBLE"


def test_weekly_bucket_label_is_not_session_evidence():
    rows = family()
    rows[1]["session_evidence"] = "weekly_interval_label"
    result = diagnostic.evaluate_eligibility(pd.DataFrame(rows), {"AAPL"}, "1wk")
    assert result.loc["AAPL", "state"] == "INELIGIBLE"
    assert "price-bound" in result.loc["AAPL", "reason"]


def test_unavailable_persisted_evidence_is_unknown():
    daily = pd.Series(row(interval="1d", revision_ambiguous=np.nan))
    state, reason = diagnostic._eligibility(daily, "1d")
    assert state == diagnostic.UNKNOWN
    assert "revision ambiguity" in reason


def analyze_rows(rows):
    intervals = diagnostic.validate_manifest(pd.DataFrame(rows))
    affected = diagnostic.select_affected_monthly(intervals)
    return diagnostic.analyze(intervals, affected)


def test_dw_agreement_uses_existing_price_hint_tolerance():
    result = analyze_rows(family(w=101.004))
    detail = result.detail.loc["AAPL"]
    assert detail.dw_status == "AGREE"
    assert detail.tolerance == pytest.approx(0.005)


def test_dw_conflict():
    result = analyze_rows(family(w=101.006))
    assert result.summary["DW_CONFLICT"] == 1
    assert result.primary_reasons.loc["AAPL"] == "DW_CONFLICT"


@pytest.mark.parametrize(("candidate", "expected"), [(101.0, True), (103.0, False)])
def test_monthly_range_gate(candidate, expected):
    result = analyze_rows(family(d=candidate, w=candidate))
    assert bool(result.detail.loc["AAPL", "m_range_pass"]) is expected


def test_repairability_counts_and_primary_partition_are_deterministic():
    rows = family(
        terminal_session="2026-10-01",
        session_evidence="yfinance_interval_merge_lastTrade",
    )
    result = analyze_rows(rows)
    assert result.summary["STRICT_R3_REPAIRABLE"] == 1
    assert result.summary["STRICT_R3_NOT_REPAIRABLE"] == 0
    assert result.primary_reasons.to_dict() == {"AAPL": "REPAIRABLE"}


def test_provider_metadata_parsing():
    value = json.dumps({"lastTrade": {"Price": 101, "Time": "2026-10-01T16:00:00-04:00"}})
    metadata = diagnostic.parse_metadata(value)
    assert metadata["lastTrade"]["Price"] == 101
    assert diagnostic._date_value(metadata["lastTrade"]["Time"]).isoformat() == "2026-10-01"
    assert diagnostic.parse_metadata("not-json") == {}


def test_survivor_date_mapping_and_stale_period(capsys, monkeypatch):
    monkeypatch.setattr(diagnostic, "SURVIVORS", frozenset({"OCT", "SEP", "AUG"}))
    intervals = pd.DataFrame([
        row("OCT", "1mo"), row("SEP", "1mo"), row("AUG", "1mo")
    ])
    snapshot = pd.DataFrame(
        {"symbol": ["OCT", "SEP", "AUG"]},
        index=pd.to_datetime(["2026-10-01", "2026-09-01", "2026-08-01"]),
    )
    diagnostic.print_survivors(intervals, snapshot)
    output = capsys.readouterr().out
    assert '"symbol": "OCT"' in output and '"belongs_to_october_2026": true' in output
    assert output.count('"stale_provider_explanation": "D_EVIDENCE_INSUFFICIENT"') == 2


def test_stale_monthly_period_classification():
    result = analyze_rows(family(terminal_session="2026-09-30", session_evidence="old"))
    assert result.detail.loc["AAPL", "m_current_period"] == "FALSE"
    assert result.primary_reasons.loc["AAPL"] == "STALE_M_PERIOD"


def test_missing_columns_fail_clearly():
    with pytest.raises(diagnostic.DiagnosticFailure, match="missing required columns.*raw_close"):
        diagnostic.validate_manifest(pd.DataFrame(family()).drop(columns="raw_close"))


def test_diagnostic_uses_reads_only_and_never_calls_provider(monkeypatch, capsys):
    rows = family(terminal_session="2026-10-01", session_evidence="evidence")
    manifest = pd.DataFrame(rows)
    monkeypatch.setattr(diagnostic, "SURVIVORS", frozenset({"AAPL"}))
    snapshot = pd.DataFrame({"symbol": ["AAPL"]}, index=pd.to_datetime(["2026-10-01"]))

    def load(path):
        return manifest if str(path).endswith("manifest.parquet") else snapshot

    def forbidden(*args, **kwargs):
        pytest.fail("write/provider API invoked")

    monkeypatch.setattr(storage, "load_parquet", load)
    for name in ("save_parquet", "compare_and_swap_bytes", "create_bytes_if_absent", "delete_s3_prefix"):
        monkeypatch.setattr(storage, name, forbidden)
    # The diagnostic does not import sources/yfinance; this sentinel proves a
    # future accidental provider import/call cannot hide behind the loader.
    import etl.sources
    monkeypatch.setattr(etl.sources, "safe_load_eod", forbidden)

    result = diagnostic.run_diagnostic(data_root=Path("/read-only"))
    assert result.summary["TOTAL_AFFECTED_M"] == 1
    output = capsys.readouterr().out
    assert "[PRE_NORMALIZATION_LIMITATION]" in output
    for field in (
        "LEGACY_OPEN_GATED_R3_REPAIRABLE", "LEGACY_OPEN_GATED_R3_NOT_REPAIRABLE",
        "CLOSE_ONLY_R3_REPAIRABLE", "CLOSE_ONLY_R3_NOT_REPAIRABLE",
        "OPEN_ONLY_BLOCKED_CLOSE_REPAIR", "OPEN_BELOW_LOW", "OPEN_ABOVE_HIGH",
        "OPEN_NONFINITE", "LOW_ABOVE_HIGH", "CANDIDATE_CLOSE_OUTSIDE_LOW_HIGH",
        "OPEN_OUTSIDE_LOW_HIGH",
    ):
        assert f"{field}=" in output


def test_workflow_is_manual_only_read_only_and_has_root_import_contract():
    text = WORKFLOW.read_text()
    trigger = text.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    assert not any(name in trigger for name in ("schedule:", "push:", "pull_request:"))
    assert "permissions:\n  contents: read" in text
    assert "DATA_BACKEND: s3" in text and "S3_BUCKET_DATA: stock-intel-data-prod" in text
    assert "PYTHONPATH=. python diagnostics/diagnose_stock_eod_monthly_collapse.py" in text
    assert not any(name in text for name in (
        "run_stock_eod_family", "run_timeframe", "run_combo", "yfinance", "save_parquet",
        "compare_and_swap", "create_bytes", "registry",
    ))


@pytest.mark.parametrize('opening', [98.0, 100.0, 103.0])
@pytest.mark.parametrize('gate', ['pass', 'd_missing', 'w_missing', 'd_quality', 'w_quality',
                                  'd_adjustment', 'w_session', 'w_evidence', 'conflict',
                                  'period', 'currentness', 'bounds', 'candidate',
                                  'volume', 'revision'])
def test_open_counterfactual_changes_only_open_geometry(opening, gate):
    rows = family(raw_open=opening, terminal_session='2026-10-01', session_evidence='evidence')
    if gate == 'd_missing':
        rows = rows[1:]
    elif gate == 'w_missing':
        rows.pop(1)
    elif gate in {'d_quality', 'w_quality'}:
        rows[0 if gate == 'd_quality' else 1]['quality_status'] = 'invalid'
    elif gate == 'd_adjustment':
        rows[0]['adjustment_mode'] = 'adjusted'
    elif gate == 'w_session':
        rows[1]['terminal_session'] = '2026-09-30'
    elif gate == 'w_evidence':
        rows[1]['session_evidence'] = 'weekly_interval_label'
    elif gate == 'conflict':
        rows[1]['raw_close'] = 101.006
    elif gate == 'period':
        rows[2]['terminal_session'] = '2026-09-30'
    elif gate == 'currentness':
        rows[2]['session_evidence'] = None
    elif gate == 'bounds':
        rows[2].update(raw_low=102.0, raw_high=99.0)
    elif gate == 'candidate':
        rows[0]['raw_close'] = rows[1]['raw_close'] = 104.0
    elif gate == 'volume':
        rows[2]['raw_volume'] = -1.0
    elif gate == 'revision':
        rows[0]['revision_ambiguous'] = True
    result = analyze_rows(rows)
    legacy = gate == 'pass' and 99 <= opening <= 102
    close_only = gate == 'pass'
    detail = result.detail.loc['AAPL']
    assert bool(detail.strict_r3_repairable) == legacy
    assert bool(detail.legacy_open_gated_r3_repairable) == legacy
    assert bool(detail.close_only_r3_repairable) == close_only
    assert bool(detail.open_only_blocked_close_repair) == (close_only and not legacy)
    assert result.summary['CLOSE_ONLY_R3_REPAIRABLE'] == (
        result.summary['LEGACY_OPEN_GATED_R3_REPAIRABLE']
        + result.summary['OPEN_ONLY_BLOCKED_CLOSE_REPAIR']
    )
    if gate == 'bounds':
        assert result.summary['LOW_ABOVE_HIGH'] == 1
        assert result.summary['OPEN_ONLY_BLOCKED_CLOSE_REPAIR'] == 0
    if gate == 'candidate':
        assert result.summary['CANDIDATE_CLOSE_OUTSIDE_LOW_HIGH'] == 1


@pytest.mark.parametrize('opening', [np.nan, np.inf, -np.inf])
def test_nonfinite_open_still_excluded_from_exact_population_and_counterfactual(opening):
    intervals = diagnostic.validate_manifest(pd.DataFrame(family(
        raw_open=opening, terminal_session='2026-10-01', session_evidence='evidence',
    )))
    with pytest.raises(diagnostic.DiagnosticFailure, match='population mismatch'):
        diagnostic.select_affected_monthly(intervals)
    # Exercise the defensive substrate gate without weakening production selection.
    affected = intervals.loc[intervals.interval.eq('1mo')]
    result = diagnostic.analyze(intervals, affected)
    assert result.summary['OPEN_NONFINITE'] == 1
    assert result.summary['CLOSE_ONLY_R3_REPAIRABLE'] == 0
    assert result.summary['LEGACY_OPEN_GATED_R3_REPAIRABLE'] == 0


@pytest.mark.parametrize('interval_index', [0, 1])
def test_finalized_or_repaired_peer_close_is_never_used(interval_index):
    rows = family(raw_open=98.0, terminal_session='2026-10-01', session_evidence='evidence')
    rows[interval_index].update(raw_close=np.nan, finalized_close=101.0,
                                canonical_observed_close=101.0,
                                finalization_status='repaired')
    result = analyze_rows(rows)
    assert result.summary['CLOSE_ONLY_R3_REPAIRABLE'] == 0
    assert result.summary['OPEN_ONLY_BLOCKED_CLOSE_REPAIR'] == 0
    assert result.detail.candidate.isna().all()


def test_comparative_population_identity_and_geometry_counts(monkeypatch):
    rows = []
    for symbol, opening, low, high in [('IN', 100, 99, 102), ('BELOW', 98, 99, 102),
                                      ('ABOVE', 103, 99, 102), ('INVERTED', 100, 102, 99)]:
        rows.extend(family(symbol, raw_open=opening, raw_low=low, raw_high=high,
                           terminal_session='2026-10-01', session_evidence='evidence'))
    monkeypatch.setattr(diagnostic, 'EXPECTED_AFFECTED', 4)
    summary = analyze_rows(rows).summary
    assert summary['LEGACY_OPEN_GATED_R3_REPAIRABLE'] == 1
    assert summary['CLOSE_ONLY_R3_REPAIRABLE'] == 3
    assert summary['OPEN_ONLY_BLOCKED_CLOSE_REPAIR'] == 2
    assert summary['LEGACY_OPEN_GATED_R3_NOT_REPAIRABLE'] == 3
    assert summary['CLOSE_ONLY_R3_NOT_REPAIRABLE'] == 1
    assert summary['OPEN_BELOW_LOW'] == 2  # includes inverted bounds, separately counted
    assert summary['OPEN_ABOVE_HIGH'] == 2
    assert summary['LOW_ABOVE_HIGH'] == 1
