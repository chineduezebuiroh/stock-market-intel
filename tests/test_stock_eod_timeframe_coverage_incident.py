from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from core import storage
from core import stock_eod_family as family
from diagnostics import stock_eod_timeframe_coverage_incident as d

RUN = d.LATEST


def fixture_generation(root, members=None, run_id=RUN):
    members = members or {"daily": {"A", "COF"}, "weekly": {"A"}, "monthly": {"A", "COF", "V"}}
    base = family.manifest_path(run_id, root).parent
    base.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for tf, symbols in members.items():
        frame = pd.DataFrame({"symbol": sorted(symbols), "close": 10.0})
        buf = BytesIO()
        frame.to_parquet(buf)
        payload = buf.getvalue()
        (base / (tf + ".parquet")).write_bytes(payload)
        artifacts[tf] = dict(filename=tf + ".parquet", sha256=sha256(payload).hexdigest(), byte_size=len(payload), row_count=len(frame), symbol_count=len(frame))
    manifest = dict(schema_version=family.SCHEMA_VERSION, producer=family.PRODUCER, family_run_id=run_id, staged_at_utc="2026-10-13T00:00:00Z", artifacts=artifacts)
    (base / "manifest.json").write_text(json.dumps(manifest))
    telemetry = []
    universe = sorted(set.union(*members.values()))
    for index, symbol in enumerate(universe):
        for interval, tf in d.INTERVALS.items():
            accepted = symbol in members[tf]
            telemetry.append(dict(record_type="interval", family_run_id=run_id, symbol=symbol, interval=interval,
                timeframe=tf, fetch_started_at=f"2026-10-{'12T23:59' if index == 0 else '13T00:01'}:{index:02d}Z",
                acquisition_status="success", quality_status="valid" if accepted else "invalid",
                quality_reason=None if accepted else "invalid terminal required fields: close",
                finalization_status="valid" if accepted else "invalid", processing_status="accepted" if accepted else "excluded",
                processing_reason=None if accepted else "finalized observation is not eligible",
                reconciliation_status="not_needed", raw_open=9, raw_high=11, raw_low=8, raw_close=10 if accepted else None,
                raw_adj_close=10 if accepted else None, raw_volume=100, repair_provenance_json='{"status":"native_unmodified"}'))
    telemetry.append(dict(record_type="run_summary", family_run_id=run_id, routed_symbol_count=len(universe)))
    path = root / "_health" / "stocks_eod_family" / run_id / "manifest.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(telemetry).to_parquet(path)
    return members, path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DATA_BACKEND", "local")
    root = tmp_path / "data"
    members, telemetry = fixture_generation(root)
    return root, members, telemetry, tmp_path / "output"


def run(evidence, **kwargs):
    root, _, _, output = evidence
    return d.run_diagnostic(data_root=root, output_dir=output, run_ids=[RUN], expectations={RUN: (3, 2, 1, 3, 1)}, **kwargs)


def test_complete_missing_sets_reason_reconciliation_and_focus(evidence):
    reports, boundary = run(evidence)
    missing = reports["family_missing_symbols"]
    assert {tf: set(missing.loc[missing.timeframe == tf].symbol) for tf in [*d.TIMEFRAMES, "intersection"]} == {
        "daily": {"V"}, "weekly": {"COF", "V"}, "monthly": set(), "intersection": {"COF", "V"}}
    partition = reports["missing_reason_partition"]
    assert partition.groupby("timeframe")["count"].sum().to_dict() == {"daily": 1, "weekly": 2}
    focus = reports["focus_symbol_longitudinal"]
    cof = focus.loc[focus.symbol == "COF"]
    assert len(cof) == 3
    assert cof.set_index("timeframe").immutable_artifact_member.to_dict() == {"daily": True, "weekly": False, "monthly": True}
    assert cof.set_index("timeframe").loc["weekly", "raw_terminal_label"] == d.UNKNOWN
    assert boundary["verdict"] == "CONFIRMED"
    assert boundary["acquisition_suffix"] is True
    assert len(reports["oct13_weekly_missing_detail"]) == 2
    assert reports["oct13_weekly_missing_detail"].set_index("symbol").daily_survived.to_dict() == {"COF": True, "V": False}
    assert len(list(evidence[3].glob("*.csv"))) == 10
    assert (evidence[3] / "incident_summary.md").exists()


def test_missing_stage_columns_are_unknown_not_membership_inference(evidence):
    path = evidence[2]
    frame = pd.read_parquet(path).drop(columns=["acquisition_status", "quality_status", "finalization_status", "processing_status", "reconciliation_status"])
    frame.to_parquet(path)
    reports, _ = run(evidence)
    stages = reports["family_stage_counts"]
    assert set(stages.loc[stages.metric == "acquisition_success", "count"]) == {d.UNKNOWN}
    assert set(stages.loc[stages.metric == "finalized_valid", "count"]) == {d.UNKNOWN}
    assert set(stages.loc[stages.metric == "immutable_artifact_membership", "count"]) == {1, 2, 3}


@pytest.mark.parametrize("row,category", [
    ({"acquisition_status": "failed", "acquisition_error": "empty provider payload"}, "acquisition_empty_unavailable_error"),
    ({"quality_reason": "invalid terminal required fields: close"}, "missing_or_nonfinite_required_fields"),
    ({"raw_close": float("inf")}, "nonfinite_required_fields"),
    ({"raw_volume": -1}, "negative_invalid_volume"),
    ({"reconciliation_status": "conflict", "quality_reason": "invalid terminal required fields: close"}, "reconciliation_conflict"),
    ({"reconciliation_status": "excluded"}, "reconciliation_unavailable"),
    ({"repair_provenance_json": '{"reason":"canonical close violates target high/low"}'}, "repair_bounds_failure"),
    ({"processing_status": "rejected", "processing_reason": "indicator output is empty"}, "downstream_processing_rejection"),
    ({"processing_reason": "stale session mismatch"}, "stale_session_current_period_failure"),
    ({"finalization_status": "invalid"}, "finalized_observation_ineligible"),
    ({}, "unknown"),
])
def test_decisive_reason_categories(row, category):
    assert d.reason(pd.Series(row))[0] == category


@pytest.mark.parametrize("times,missing,verdict,mixed,present_after", [
    (["2026-10-12T23:59:00Z", "2026-10-13T00:01:00Z", "2026-10-13T00:02:00Z"], {"B", "C"}, "CONFIRMED", False, 0),
    (["2026-10-12T23:59:00Z", "2026-10-13T00:01:00Z", "2026-10-13T00:02:00Z"], {"A", "C"}, "REJECTED", True, 1),
    (["2026-10-12T23:59:00Z", "2026-10-13T00:01:00Z", "2026-10-13T00:02:00Z"], {"B"}, "REJECTED", False, 1),
    ([None, "2026-10-13T00:01:00", "2026-10-13T00:02:00Z"], {"B", "C"}, "UNRESOLVED", False, 0),
])
def test_midnight_boundary(times, missing, verdict, mixed, present_after):
    rows = pd.DataFrame(dict(symbol=list("ABC"), timeframe="weekly", fetch_started_at=times,
                             acquisition_order=[1, 2, 3], telemetry_symbol_order=[1, 2, 3]))
    counts, result = d.boundary_analysis(RUN, rows, missing)
    assert result["verdict"] == verdict
    assert result["mixed_losses"] == mixed
    assert result["successful_after_midnight"] == present_after
    assert sum(r["count"] for r in counts if r["timestamp_basis"] == "fetch_started_at") == 3
    assert sum(r["count"] for r in counts if r["timestamp_basis"] == "fetch_completed_at" and r["period"] == "unknown") == 3


def test_overlap_jaccard():
    result = d.overlaps({"a": {"COF", "V"}, "b": {"V", "QQQ"}, "c": set(), "d": set()})
    assert result[0]["jaccard"] == pytest.approx(1 / 3)
    assert result[0]["overlap"] == 1
    assert json.loads(result[0]["recurring_symbols"]) == ["V"]
    assert result[1]["jaccard"] == 0
    assert result[2]["jaccard"] == 1


@pytest.mark.parametrize("corruption", ["hash", "row_count", "symbol_count", "expectation", "family_identity", "duplicate", "incomplete"])
def test_evidence_mismatch_fails_closed(evidence, corruption):
    root, _, telemetry, output = evidence
    path = family.manifest_path(RUN, root)
    if corruption == "hash":
        artifact = family.artifact_path(RUN, "weekly", root)
        data = bytearray(artifact.read_bytes()); data[len(data) // 2] ^= 1
        artifact.write_bytes(data)
    elif corruption in {"row_count", "symbol_count"}:
        manifest = json.loads(path.read_text())
        manifest["artifacts"]["weekly"][corruption] = 99
        path.write_text(json.dumps(manifest))
    elif corruption == "expectation":
        with pytest.raises(d.DiagnosticFailure, match="retained-log"):
            d.run_diagnostic(data_root=root, output_dir=output, run_ids=[RUN], expectations={RUN: (3, 3, 3, 3, 3)})
        assert not output.exists()
        return
    else:
        frame = pd.read_parquet(telemetry)
        if corruption == "family_identity": frame.loc[0, "family_run_id"] = "wrong"
        if corruption == "duplicate": frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        if corruption == "incomplete": frame = frame.drop(index=0)
        frame.to_parquet(telemetry)
    with pytest.raises(d.DiagnosticFailure): run(evidence)
    assert not output.exists()


def test_no_storage_writes_provider_calls_or_network(evidence, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic invoked a mutation or provider/network API")
    for name in ["save_parquet", "compare_and_swap_bytes", "create_bytes_if_absent", "delete_s3_prefix"]:
        monkeypatch.setattr(storage, name, forbidden)
    import socket
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    # Block providers if installed, without making them a diagnostic dependency.
    import etl.sources
    import jobs.run_stock_eod_family
    monkeypatch.setattr(etl.sources, "safe_load_eod", forbidden)
    monkeypatch.setattr(etl.sources, "load_eod", forbidden)
    monkeypatch.setattr(jobs.run_stock_eod_family, "run_family", forbidden)
    run(evidence)


def test_output_cannot_overwrite_production_data(evidence):
    root = evidence[0]
    with pytest.raises(d.DiagnosticFailure, match="outside"):
        d.run_diagnostic(data_root=root, output_dir=root / "reports", run_ids=[RUN])


def test_workflow_is_manual_read_only_and_has_minimal_dependencies():
    path = Path(__file__).resolve().parents[1] / ".github/workflows/stock-eod-timeframe-coverage-incident-diagnostic.yml"
    raw = path.read_text()
    config = yaml.safe_load(raw)
    # PyYAML YAML 1.1 interprets `on` as True.
    assert set(config.get("on", config.get(True))) == {"workflow_dispatch"}
    assert config["permissions"] == {"contents": "read"}
    assert "actions/upload-artifact@v4" in raw
    assert "python-version: \"3.11\"" in raw
    assert "continue-on-error" not in raw
    assert "yfinance" not in raw
    assert "run_stock_eod_family.py" not in raw
    assert "stock-intel-data-prod" in raw


def test_healthy_generation_can_emit_empty_loss_reports(evidence):
    root, _, _, output = evidence
    fixture_generation(root, {tf: {"A", "COF", "V"} for tf in d.TIMEFRAMES})
    reports, _ = d.run_diagnostic(data_root=root, output_dir=output, run_ids=[RUN], expectations={RUN: (3, 3, 3, 3, 3)})
    assert reports["family_missing_symbols"].empty
    assert reports["missing_reason_partition"].empty
    assert "No rows." in (output / "incident_summary.md").read_text()


def test_daily_reasons_stable_and_vary():
    records = [dict(family_run_id=family_id, timeframe="daily", symbol=symbol,
                    reason_category="reconciliation_conflict", supporting_reasons=support,
                    fetch_started_at="2026-10-13T00:00:00Z", acquisition_order=1)
               for family_id, symbol, support in [("a", "COF", "x"), ("b", "COF", "x"), ("a", "V", "x"), ("b", "V", "y")]]
    recurrence, timing = d.daily_analysis(records)
    assert {row["symbol"]: row["reason_stability"] for row in recurrence} == {"COF": "STABLE", "V": "VARIES"}
    assert sum(row["count"] for row in timing) == 4


def test_partial_timestamps_do_not_hide_known_midnight_counterexample():
    rows = pd.DataFrame(dict(symbol=["A", "B"], timeframe="weekly", fetch_started_at=[None, "2026-10-12T23:59:00Z"],
                             acquisition_order=d.UNKNOWN, telemetry_symbol_order=[1, 2]))
    _, result = d.boundary_analysis(RUN, rows, {"B"})
    assert result["verdict"] == "REJECTED"
    assert result["unknown_timestamps"] == 1


def test_repaired_observation_downstream_failure_not_blame_raw_close():
    row = pd.Series(dict(acquisition_status="success", quality_reason="invalid terminal required fields: close",
                         finalization_status="repaired", processing_status="rejected", processing_reason="indicator output is empty"))
    assert d.reason(row)[0] == "downstream_processing_rejection"


def test_s3_reader_only_uses_get_object_on_fixed_evidence_paths(evidence, monkeypatch):
    root, _, _, _ = evidence
    objects = {"data/" + str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    reads = []
    class ReadOnlyClient:
        def get_object(self, *, Bucket, Key):
            assert Bucket == "test-production-evidence"
            assert Key in objects
            assert "current.json" not in Key
            reads.append(Key)
            return {"Body": BytesIO(objects[Key]), "ETag": "test"}
        def __getattr__(self, name):
            raise AssertionError(f"forbidden S3 operation: {name}")
    monkeypatch.setattr(storage, "_DATA_BACKEND", "s3")
    monkeypatch.setattr(storage, "DATA", root)
    monkeypatch.setattr(storage, "_s3_client", lambda: ReadOnlyClient())
    monkeypatch.setenv("S3_BUCKET_DATA", "test-production-evidence")
    monkeypatch.setenv("S3_PREFIX_DATA", "data")
    run(evidence)
    assert set(reads) == set(objects)


def test_cli_integrity_failure_returns_nonzero_without_reports(monkeypatch, tmp_path):
    def fail(**kwargs): raise d.DiagnosticFailure("bad evidence")
    monkeypatch.setattr(d, "run_diagnostic", fail)
    assert d.main(["--output-dir", str(tmp_path / "reports")]) == 1
    assert not (tmp_path / "reports").exists()
