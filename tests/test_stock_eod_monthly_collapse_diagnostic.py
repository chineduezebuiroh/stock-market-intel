from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import storage
from diagnostics import diagnose_stock_eod_monthly_collapse as diagnostic


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "stock-eod-monthly-collapse-diagnostic.yml"


def manifest_row(**overrides) -> dict:
    row = {
        "record_type": "interval",
        "interval": "1mo",
        "symbol": "AAPL",
        "terminal_session": "2026-10-01",
        "session_evidence": "interval_label",
        "acquisition_status": "success",
        "acquisition_error": None,
        "quality_status": "valid",
        "quality_reason": None,
        "close_is_null": False,
        "raw_open": 100.0,
        "raw_high": 102.0,
        "raw_low": 99.0,
        "raw_close": 101.0,
        "raw_adj_close": 101.0,
        "raw_volume": 1_000.0,
        "finalization_status": "valid",
        "processing_status": "accepted",
        "processing_reason": None,
    }
    row.update(overrides)
    return row


def malformed_row(**overrides) -> dict:
    values = {
        "quality_status": "invalid",
        "quality_reason": "invalid terminal required fields: close",
        "close_is_null": True,
        "raw_close": np.nan,
        "finalization_status": "invalid",
        "processing_status": "excluded",
        "processing_reason": "finalized observation is not eligible",
    }
    values.update(overrides)
    return manifest_row(**values)


def test_manifest_filtering_selects_only_monthly_interval_rows():
    rows = [
        manifest_row(symbol="AAPL"),
        manifest_row(symbol="MSFT", interval="1d"),
        manifest_row(symbol="NVDA", record_type="run_summary"),
    ]
    result = diagnostic.filter_monthly_intervals(pd.DataFrame(rows))
    assert result["symbol"].tolist() == ["AAPL"]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([malformed_row()], "CONFIRMED_NULL_RAW_CLOSE_BOUNDARY_SHAPE"),
        (
            [
                manifest_row(
                    acquisition_status="failed",
                    acquisition_error="empty provider payload",
                    quality_status="unavailable",
                    quality_reason="empty provider payload",
                    finalization_status="excluded",
                    processing_status="excluded",
                )
            ],
            "CONFIRMED_PROVIDER_ACQUISITION_FAILURE",
        ),
        (
            [
                malformed_row(
                    raw_close=101.0,
                    close_is_null=False,
                    raw_volume=np.nan,
                    quality_reason="invalid terminal required fields: volume",
                )
            ],
            "CONFIRMED_OTHER_REQUIRED_FIELD_FAILURE",
        ),
        (
            [manifest_row(processing_status="rejected", processing_reason="indicator output is empty")],
            "CONFIRMED_POST_ACQUISITION_PROCESSING_FAILURE",
        ),
        ([manifest_row()], "INCONCLUSIVE"),
    ],
)
def test_grouped_classification_logic(rows, expected):
    counts = diagnostic.failure_counts(pd.DataFrame(rows))
    assert diagnostic.classify_failure(counts) == expected


def test_null_raw_close_pattern_requires_adj_close_and_all_other_fields():
    frame = pd.DataFrame(
        [
            malformed_row(symbol="MATCH"),
            malformed_row(symbol="NO_ADJ", raw_adj_close=np.nan),
            malformed_row(symbol="NO_HIGH", raw_high=np.nan),
            malformed_row(symbol="HAS_CLOSE", raw_close=101.0, close_is_null=False),
        ]
    )
    assert frame.loc[diagnostic.null_raw_close_pattern(frame), "symbol"].tolist() == ["MATCH"]


def test_multiple_observed_failure_categories_are_mixed():
    frame = pd.DataFrame(
        [
            malformed_row(symbol="CLOSE"),
            malformed_row(
                symbol="VOLUME",
                raw_close=101.0,
                close_is_null=False,
                raw_volume=np.nan,
                quality_reason="invalid terminal required fields: volume",
            ),
        ]
    )
    assert diagnostic.classify_failure(diagnostic.failure_counts(frame)) == "MIXED_FAILURE_MODES"


def test_missing_required_columns_fail_clearly():
    frame = pd.DataFrame([manifest_row()]).drop(columns="raw_close")
    with pytest.raises(diagnostic.DiagnosticFailure, match="missing required columns.*raw_close"):
        diagnostic.filter_monthly_intervals(frame)


def test_empty_monthly_population_fails_clearly():
    frame = pd.DataFrame([manifest_row(interval="1d")])
    with pytest.raises(diagnostic.DiagnosticFailure, match="no monthly interval rows"):
        diagnostic.filter_monthly_intervals(frame)


def test_run_is_read_only_and_snapshot_comparison_is_secondary(monkeypatch, capsys):
    manifest = pd.DataFrame([malformed_row(), manifest_row(symbol="SURVIVOR")])

    def load(path):
        if str(path).endswith("manifest.parquet"):
            return manifest
        raise FileNotFoundError(path)

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic invoked a storage write API")

    monkeypatch.setattr(storage, "load_parquet", load)
    monkeypatch.setattr(storage, "save_parquet", forbidden)
    monkeypatch.setattr(storage, "compare_and_swap_bytes", forbidden)
    monkeypatch.setattr(storage, "create_bytes_if_absent", forbidden)
    monkeypatch.setattr(storage, "delete_s3_prefix", forbidden)

    result = diagnostic.run_diagnostic(data_root=Path("/read-only-data"))

    output = capsys.readouterr().out
    assert result == "CONFIRMED_NULL_RAW_CLOSE_BOUNDARY_SHAPE"
    assert "[MONTHLY] total_interval_rows=2" in output
    assert "[SNAPSHOT_COMPARISON_UNAVAILABLE]" in output
    assert "[CLASSIFICATION] CONFIRMED_NULL_RAW_CLOSE_BOUNDARY_SHAPE" in output


def test_workflow_is_manual_only_read_only_and_uses_repo_root_import_contract():
    workflow = WORKFLOW.read_text()
    trigger_block = workflow.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger_block
    assert "schedule:" not in trigger_block
    assert "push:" not in trigger_block
    assert "pull_request:" not in trigger_block
    assert "permissions:\n  contents: read" in workflow
    assert "DATA_BACKEND: s3" in workflow
    assert "S3_BUCKET_DATA: stock-intel-data-prod" in workflow
    assert "PYTHONPATH=. python diagnostics/diagnose_stock_eod_monthly_collapse.py" in workflow
    forbidden_terms = (
        "save_parquet",
        "compare_and_swap",
        "create_bytes",
        "run_stock_eod_family.py",
        "run_combo.py",
        "run_timeframe.py",
        "yfinance",
    )
    assert not any(term in workflow for term in forbidden_terms)
