from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from diagnostics.validate_eod_family_shadow import validate_manifest
from etl.eod_family import MANIFEST_SCHEMA


def manifest(symbol="AAPL"):
    rows = []
    for interval in ("1d", "1wk", "1mo"):
        row = {column: None for column in MANIFEST_SCHEMA}
        row.update(
            record_type="interval", family_run_id="run", symbol=symbol,
            interval=interval, acquisition_status="success", quality_status="valid",
            close_is_null=False, raw_open=99.0, raw_high=101.0, raw_low=98.0,
            raw_close=100.0, raw_adj_close=99.5, raw_volume=1000.0,
            finalized_open=99.0, finalized_high=101.0, finalized_low=98.0,
            finalized_close=100.0, finalized_adj_close=99.5, finalized_volume=1000.0,
            repair_provenance_json=json.dumps({"status": "native_unmodified"}),
            reconciliation_status="not_needed", requests_attempted=3,
            requests_succeeded=3, requests_failed=0,
            reconciliation_provider_requests=0,
            validity_pattern="1d:valid/1wk:valid/1mo:valid",
            close_null_pattern="1d:close_null=false/1wk:close_null=false/1mo:close_null=false",
        )
        rows.append(row)
    summary = {column: None for column in MANIFEST_SCHEMA}
    summary.update(
        record_type="run_summary", family_run_id="run", requests_attempted=3,
        requests_succeeded=3, requests_failed=0, reconciliation_provider_requests=0,
    )
    rows.append(summary)
    return pd.DataFrame(rows, columns=MANIFEST_SCHEMA)


def test_validator_accepts_complete_native_family():
    result = validate_manifest(manifest(), ["AAPL"])
    assert result["needs_indicators"] is True
    assert len(result["intervals"]) == 3


def test_validator_accepts_close_only_daily_repair_and_preserves_native():
    value = manifest()
    daily = value["interval"].eq("1d")
    value.loc[daily, ["quality_status", "quality_reason", "raw_close", "close_is_null"]] = [
        "invalid", "invalid terminal required fields: close", float("nan"), True,
    ]
    value.loc[daily, "finalized_close"] = 100.0
    value.loc[daily, "reconciliation_status"] = "canonical"
    value.loc[daily, "reconciliation_tier"] = "tier_1_independent_consensus"
    value.loc[daily, "canonical_source_intervals"] = "1mo+1wk"
    value.loc[daily, "terminal_session"] = "2026-09-23"
    value.loc[value["interval"].isin(["1wk", "1mo"]), "terminal_session"] = "2026-09-23"
    value.loc[daily, "repair_provenance_json"] = json.dumps({"data_quality_status": "repaired"})

    result = validate_manifest(value, ["AAPL"])
    assert result["needs_indicators"] is True
    assert pd.isna(result["intervals"].loc[daily, "raw_close"].iloc[0])
    assert result["intervals"].loc[daily, "finalized_close"].iloc[0] == 100.0


@pytest.mark.parametrize("mutation", ["extra_request", "wm_repair", "provider_reconcile"])
def test_validator_rejects_hard_invariant_violations(mutation):
    value = manifest()
    if mutation == "extra_request":
        value.loc[value["record_type"].eq("interval"), "requests_attempted"] = 4
    elif mutation == "wm_repair":
        value.loc[value["interval"].eq("1wk"), "finalized_close"] = 101.0
    else:
        value.loc[value["record_type"].eq("interval"), "reconciliation_provider_requests"] = 1
    with pytest.raises(AssertionError):
        validate_manifest(value, ["AAPL"])


def test_validator_allows_explicit_isolated_empty_interval():
    value = manifest()
    weekly = value["interval"].eq("1wk")
    value.loc[weekly, "acquisition_status"] = "failed"
    value.loc[weekly, "acquisition_error"] = "empty provider payload"
    value.loc[weekly, "quality_status"] = "unavailable"
    for prefix in ("raw", "finalized"):
        for field in ("open", "high", "low", "close", "adj_close", "volume"):
            value.loc[weekly, f"{prefix}_{field}"] = float("nan")
    value.loc[value["record_type"].eq("interval"), ["requests_succeeded", "requests_failed"]] = [2, 1]
    value.loc[value["record_type"].eq("run_summary"), ["requests_succeeded", "requests_failed"]] = [2, 1]
    result = validate_manifest(value, ["AAPL"])
    assert len(result["intervals"]) == 3


def test_live_workflow_is_manual_bounded_and_shadow_only():
    workflow = Path(".github/workflows/eod_family_shadow.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "- tiny" in workflow and "- seven" in workflow
    assert "schedule:" not in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "jobs/run_stock_eod_family_shadow.py" in workflow
    assert "diagnostics/validate_eod_family_shadow.py" in workflow
    for forbidden in (
        "run_stocks_eod_guarded.py", "jobs/run_timeframe.py", "jobs/run_combo.py",
        "execution_registry", "notify", "--cascade", "--no-indicators",
    ):
        assert forbidden not in workflow
