from __future__ import annotations

import argparse
import ast
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from diagnostics.investigate_scoring_lineage import (
    INDICATORS,
    compare_close_lineage,
    compare_indicator_lineage,
    filter_as_of,
    local_output_dir,
    run,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "diagnostics" / "investigate_scoring_lineage.py"
WORKFLOW = ROOT / ".github" / "workflows" / "investigate_scoring_lineage.yml"


class EmptyPaginator:
    def paginate(self, **_kwargs):
        return [{}]


class ReadOnlyEmptyClient:
    calls: list[str]

    def __init__(self):
        self.calls = []

    def get_paginator(self, operation):
        self.calls.append(operation)
        assert operation in {"list_objects_v2", "list_object_versions"}
        return EmptyPaginator()

    def get_object(self, **_kwargs):
        self.calls.append("get_object")
        raise AssertionError("no object should be read when history is unavailable")


def test_as_of_filter_excludes_every_post_market_date_bar():
    frame = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-09-20", "2026-09-21", "2026-09-22"]),
    )
    result = filter_as_of(frame, date(2026, 9, 21))
    assert result.index.max() == pd.Timestamp("2026-09-21")
    assert result["close"].tolist() == [1.0, 2.0]


def test_close_lineage_identifies_first_transition():
    result = compare_close_lineage(
        "daily",
        250.0,
        "2026-09-21",
        250.0,
        "2026-09-21",
        float("nan"),
        "2026-09-21",
        canonical_available=True,
        snapshot_available=True,
        combo_available=True,
    )
    assert result["canonical_to_snapshot"] == "match"
    assert result["snapshot_to_combo"] == "mismatch"
    assert result["first_issue"] == "snapshot_to_combo"


def test_indicator_lineage_reports_matches_mismatches_and_unavailable():
    matched = compare_indicator_lineage(
        "daily",
        "wyckoff_stage",
        2,
        2,
        2,
        recomputed_available=True,
        snapshot_available=True,
        combo_available=True,
    )
    unavailable = compare_indicator_lineage(
        "daily",
        "wyckoff_stage",
        float("nan"),
        2,
        0,
        recomputed_available=False,
        snapshot_available=True,
        combo_available=True,
    )
    assert matched["recomputed_to_snapshot"] == "match"
    assert unavailable["recomputed_to_snapshot"] == "unavailable"
    assert unavailable["snapshot_to_combo"] == "mismatch"


def test_output_directory_must_remain_below_local_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert local_output_dir(Path("diagnostic_artifacts"), tmp_path).is_dir()
    with pytest.raises(ValueError, match="under local root"):
        local_output_dir(tmp_path.parent / "escape", tmp_path)


def test_missing_history_emits_unavailable_bundle_without_latest_substitution(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    client = ReadOnlyEmptyClient()
    args = argparse.Namespace(
        bucket="bucket",
        prefix="data",
        market_date="2026-09-21",
        stock_symbol="CRWD",
        primary_etf="IGV",
        secondary_etf="XLK",
        output_dir="diagnostic_artifacts",
        indicator_config=str(ROOT / "config" / "indicator_params.yaml"),
    )
    output = run(args, client=client)
    metadata = json.loads((output / "metadata.json").read_text())
    inventory = pd.read_csv(output / "artifact_inventory.csv")
    assert "F. INSUFFICIENT HISTORICAL EVIDENCE" in metadata["classifications"]
    assert inventory["evidence_status"].eq("unavailable").all()
    assert "get_object" not in client.calls
    assert not list(tmp_path.rglob("*.parquet"))


def test_timestamp_and_date_metadata_are_in_machine_readable_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        bucket="bucket",
        prefix="data",
        market_date="2026-09-21",
        stock_symbol="CRWD",
        primary_etf="IGV",
        secondary_etf="XLK",
        output_dir="diagnostic_artifacts",
        indicator_config=str(ROOT / "config" / "indicator_params.yaml"),
    )
    output = run(args, client=ReadOnlyEmptyClient())
    metadata = json.loads((output / "metadata.json").read_text())
    inventory = pd.read_csv(output / "artifact_inventory.csv")
    assert metadata["market_date"] == "2026-09-21"
    assert "generated_at_utc" in metadata
    assert {"artifact_timestamp_utc", "observation_timestamp"}.issubset(
        inventory.columns
    )


def test_known_etf_volume_mismatch_is_reported_not_repaired():
    source = SCRIPT.read_text()
    assert 'row.get("significant_volume"' not in source
    assert '"scorer_field": "significant_volume"' in source
    assert '"snapshot_field": "sig_vol_current_bar"' in source
    assert '"repaired_by_diagnostic": False' in source
    assert "sig_vol_current_bar" in INDICATORS


def test_diagnostic_ast_has_no_s3_or_storage_mutation_calls():
    tree = ast.parse(SCRIPT.read_text())
    forbidden = {
        "put_object",
        "upload_file",
        "upload_fileobj",
        "delete_object",
        "delete_objects",
        "copy_object",
        "save_parquet",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden)
    assert "core.storage" not in SCRIPT.read_text()


def test_workflow_has_production_read_configuration_and_local_upload():
    workflow_text = WORKFLOW.read_text()
    workflow = yaml.safe_load(workflow_text)
    env = workflow["jobs"]["investigate"]["env"]
    assert env["S3_BUCKET_DATA"] == "stock-intel-data-prod"
    assert env["S3_PREFIX_DATA"] == "data"
    assert env["DATA_BACKEND"] == "s3"
    assert env["AWS_DEFAULT_REGION"] == "us-east-1"
    assert "secrets.AWS_ACCESS_KEY_ID" in workflow_text
    upload = workflow["jobs"]["investigate"]["steps"][-1]
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["path"] == "diagnostic_artifacts/"


def test_workflow_inputs_and_defaults_are_forwarded_to_script():
    text = WORKFLOW.read_text()
    for name, default in {
        "market_date": "2026-09-21",
        "stock_symbol": "CRWD",
        "primary_etf": "IGV",
        "secondary_etf": "XLK",
    }.items():
        assert f"{name}:" in text
        assert f'default: "{default}"' in text
        assert f"inputs.{name}" in text


def test_workflow_does_not_invoke_production_jobs_or_s3_mutations():
    text = WORKFLOW.read_text()
    forbidden = [
        "jobs/run_timeframe.py",
        "jobs/run_combo.py",
        "jobs/run_etf_trends.py",
        "aws s3 cp",
        "aws s3 sync",
        "aws s3 rm",
    ]
    assert all(term not in text for term in forbidden)
    assert "permissions:\n  contents: read" in text


def test_expected_local_artifact_names_are_emitted_by_source():
    source = SCRIPT.read_text()
    for filename in (
        "report.md",
        "metadata.json",
        "artifact_inventory.csv",
        "etf_lineage.csv",
        "_close_lineage.csv",
        "_indicator_lineage.csv",
        "_historical_combo_row.csv",
    ):
        assert filename in source
