from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from core import storage
from core import stock_eod_family as publication


def frame(*symbols: str, value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"close": [value] * len(symbols), "symbol": list(symbols)})


def publish(root: Path, run_id="prod-20260926T020000Z-00000003", state=None, **frames):
    state = state or publication.StockEODPointerState(None, None)
    return publication.publish_stock_eod_family(
        run_id, frames.get("daily", frame("AAPL", "MSFT")),
        frames.get("weekly", frame("AAPL")), frames.get("monthly", frame("AAPL", "MSFT")),
        expected_pointer_revision=state.revision,
        expected_previous_family_run_id=state.previous_family_run_id, data_root=root,
    )


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setattr(storage, "_DATA_BACKEND", "local")


def test_real_local_publish_resolve_and_partial_symbols(tmp_path):
    published = publish(tmp_path)
    resolved = publication.resolve_current_stock_eod_family(data_root=tmp_path)
    assert resolved.pointer.family_run_id == published.pointer.family_run_id
    assert list(resolved.daily.symbol) == ["AAPL", "MSFT"]
    assert list(resolved.weekly.symbol) == ["AAPL"]
    assert resolved.manifest.artifacts["daily"].sha256


@pytest.mark.parametrize("failed_name", ["daily.parquet", "weekly.parquet", "monthly.parquet", "manifest.json"])
def test_staging_failure_leaves_old_pointer(monkeypatch, tmp_path, failed_name):
    first = publish(tmp_path)
    state = publication.capture_current_pointer_state(tmp_path)
    original = storage.create_bytes_if_absent

    def fail(path, data, **kwargs):
        if Path(path).name == failed_name:
            raise OSError("injected")
        return original(path, data, **kwargs)

    monkeypatch.setattr(storage, "create_bytes_if_absent", fail)
    with pytest.raises(OSError):
        publish(tmp_path, "prod-20260926T030000Z-00000004", state=state)
    assert publication.resolve_current_stock_eod_family(data_root=tmp_path).pointer == first.pointer


def test_validation_failure_leaves_pointer(monkeypatch, tmp_path):
    first = publish(tmp_path)
    state = publication.capture_current_pointer_state(tmp_path)
    original = storage.read_bytes

    def corrupt(path):
        data = original(path)
        return b"bad" if Path(path).name == "weekly.parquet" and "00000004" in str(path) else data

    monkeypatch.setattr(storage, "read_bytes", corrupt)
    with pytest.raises(publication.StockEODFamilyCorrupt):
        publish(tmp_path, "prod-20260926T030000Z-00000004", state=state)
    monkeypatch.setattr(storage, "read_bytes", original)
    assert publication.resolve_current_stock_eod_family(data_root=tmp_path).pointer == first.pointer


def test_concurrent_publishers_and_first_publish_race(tmp_path):
    initial = publish(tmp_path)
    shared = publication.capture_current_pointer_state(tmp_path)
    winner = publish(tmp_path, "prod-20260926T040000Z-00000005", state=shared)
    with pytest.raises(publication.ConcurrentStockEODPublication):
        publish(tmp_path, "prod-20260926T050000Z-00000006", state=shared)
    assert publication.resolve_current_stock_eod_family(data_root=tmp_path).pointer == winner.pointer

    other = tmp_path / "first"
    missing = publication.StockEODPointerState(None, None)
    first = publish(other, "prod-20260926T060000Z-00000007", state=missing)
    with pytest.raises(publication.ConcurrentStockEODPublication):
        publish(other, "prod-20260926T070000Z-00000008", state=missing)
    assert publication.resolve_current_stock_eod_family(data_root=other).pointer == first.pointer
    assert initial.pointer.family_run_id != winner.pointer.family_run_id


def test_orphan_ignored_and_pointer_read_once(monkeypatch, tmp_path):
    committed = publish(tmp_path)
    state = publication.capture_current_pointer_state(tmp_path)
    pointer = publication.current_pointer_path(tmp_path)
    monkeypatch.setattr(storage, "compare_and_swap_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("die")))
    with pytest.raises(OSError):
        publish(tmp_path, "prod-20260926T080000Z-00000009", state=state)
    calls = 0
    original = storage.read_bytes_with_revision

    def count(path):
        nonlocal calls
        if Path(path) == pointer:
            calls += 1
        return original(path)

    monkeypatch.setattr(storage, "read_bytes_with_revision", count)
    assert publication.resolve_current_stock_eod_family(data_root=tmp_path).pointer == committed.pointer
    assert calls == 1


def test_missing_corrupt_manifest_and_artifact_fail_closed(tmp_path):
    result = publish(tmp_path)
    manifest = publication.manifest_path(result.pointer.family_run_id, tmp_path)
    manifest.unlink()
    with pytest.raises(publication.StockEODFamilyCorrupt):
        publication.resolve_current_stock_eod_family(data_root=tmp_path)

    root2 = tmp_path / "corrupt-manifest"
    result = publish(root2)
    publication.manifest_path(result.pointer.family_run_id, root2).write_bytes(b"{}")
    with pytest.raises(publication.StockEODFamilyCorrupt):
        publication.resolve_current_stock_eod_family(data_root=root2)

    root3 = tmp_path / "corrupt-artifact"
    result = publish(root3)
    publication.artifact_path(result.pointer.family_run_id, "daily", root3).write_bytes(b"corrupt")
    with pytest.raises(publication.StockEODFamilyCorrupt):
        publication.resolve_current_stock_eod_family(data_root=root3)


def test_immutable_conflict_and_missing_pointer_never_uses_legacy(tmp_path):
    run_id = "prod-20260926T090000Z-0000000a"
    publish(tmp_path, run_id)
    state = publication.capture_current_pointer_state(tmp_path)
    with pytest.raises(publication.ImmutableStockEODFamilyConflict):
        publish(tmp_path, run_id, state=state, daily=frame("AAPL", value=9))
    empty = tmp_path / "empty"
    empty.mkdir()
    frame("LEGACY").to_parquet(empty / "snapshot_stocks_daily.parquet")
    with pytest.raises(publication.StockEODFamilyNotPublished):
        publication.resolve_current_stock_eod_family(data_root=empty)


def test_same_id_same_exact_staging_is_idempotent_until_pointer_cas(tmp_path):
    run_id = "prod-20260926T091000Z-0000000c"
    publish(tmp_path, run_id)
    with pytest.raises(publication.ConcurrentStockEODPublication):
        publish(tmp_path, run_id, state=publication.StockEODPointerState(None, None))


@pytest.mark.parametrize("run_id", ["run", "../prod-20260926T000000Z-00000000", "prod-20261340T000000Z-00000000", "prod-20260926T000000Z-ABCDEF12"])
def test_invalid_family_ids_rejected(tmp_path, run_id):
    with pytest.raises(publication.InvalidStockEODFamilyId):
        publish(tmp_path, run_id)


def test_local_immutable_create_and_cas(tmp_path):
    path = tmp_path / "object"
    storage.create_bytes_if_absent(path, b"one")
    with pytest.raises(storage.StoragePreconditionFailed):
        storage.create_bytes_if_absent(path, b"two")
    revision = storage.read_bytes_with_revision(path).revision
    storage.compare_and_swap_bytes(path, revision, b"two")
    with pytest.raises(storage.StoragePreconditionFailed):
        storage.compare_and_swap_bytes(path, revision, b"three")


def test_s3_conditional_put_parameters_and_412(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_DATA_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET_DATA", "bucket")
    monkeypatch.setenv("S3_PREFIX_DATA", "data")
    calls = []

    class Client:
        def put_object(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("IfMatch") == '"stale"':
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}},
                    "PutObject",
                )
            return {"ETag": '"new"'}

    monkeypatch.setattr(storage, "_s3_client", lambda: Client())
    path = tmp_path / "pointer"
    storage.compare_and_swap_bytes(path, None, b"first")
    storage.compare_and_swap_bytes(path, '"old"', b"next")
    assert calls[0]["IfNoneMatch"] == "*"
    assert calls[1]["IfMatch"] == '"old"'
    with pytest.raises(storage.StoragePreconditionFailed):
        storage.compare_and_swap_bytes(path, '"stale"', b"late")
