"""Immutable publication and single-generation resolution for stock D/W/M EOD."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import re
from pathlib import Path
from typing import Mapping

import pandas as pd

from core import storage
from core.paths import DATA

SCHEMA_VERSION = 1
PRODUCER = "stocks_eod"
TIMEFRAMES = ("daily", "weekly", "monthly")
FAMILY_ID_RE = re.compile(r"^prod-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
ROOT = Path("snapshots/stocks/eod")
CURRENT = ROOT / "current.json"


class StockEODFamilyError(RuntimeError):
    pass


class StockEODFamilyNotPublished(StockEODFamilyError):
    pass


class StockEODFamilyCorrupt(StockEODFamilyError):
    pass


class InvalidStockEODFamilyId(StockEODFamilyError):
    pass


class ImmutableStockEODFamilyConflict(StockEODFamilyError):
    pass


class ConcurrentStockEODPublication(StockEODFamilyError):
    pass


@dataclass(frozen=True)
class StockEODArtifact:
    filename: str
    sha256: str
    byte_size: int
    row_count: int
    symbol_count: int


@dataclass(frozen=True)
class StockEODFamilyManifest:
    schema_version: int
    family_run_id: str
    producer: str
    staged_at_utc: str
    artifacts: Mapping[str, StockEODArtifact]


@dataclass(frozen=True)
class StockEODFamilyPointer:
    schema_version: int
    family_run_id: str
    manifest_path: str
    manifest_sha256: str
    committed_at_utc: str
    previous_family_run_id: str | None


@dataclass(frozen=True)
class StockEODPointerState:
    revision: str | None
    previous_family_run_id: str | None


@dataclass(frozen=True)
class StockEODSnapshotFamily:
    pointer: StockEODFamilyPointer
    manifest: StockEODFamilyManifest
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame


def validate_family_run_id(value: str) -> str:
    if not isinstance(value, str) or not FAMILY_ID_RE.fullmatch(value):
        raise InvalidStockEODFamilyId(f"invalid stock EOD family_run_id: {value!r}")
    try:
        datetime.strptime(value[5:21], "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise InvalidStockEODFamilyId(f"invalid stock EOD family_run_id: {value!r}") from exc
    return value


def _generation_relative(run_id: str) -> Path:
    return ROOT / "generations" / validate_family_run_id(run_id)


def current_pointer_path(data_root: Path = DATA) -> Path:
    return data_root / CURRENT


def manifest_path(run_id: str, data_root: Path = DATA) -> Path:
    return data_root / _generation_relative(run_id) / "manifest.json"


def artifact_path(run_id: str, timeframe: str, data_root: Path = DATA) -> Path:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unsupported stock EOD timeframe: {timeframe}")
    return data_root / _generation_relative(run_id) / f"{timeframe}.parquet"


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_staged_iso(run_id: str) -> str:
    value = datetime.strptime(run_id[5:21], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def _validate_frame(frame: pd.DataFrame, timeframe: str) -> tuple[int, int]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise StockEODFamilyCorrupt(f"{timeframe} snapshot is empty")
    if "symbol" not in frame.columns:
        raise StockEODFamilyCorrupt(f"{timeframe} snapshot has no symbol column")
    raw = frame["symbol"]
    normalized = raw.astype("string").str.strip().str.upper()
    if raw.isna().any() or normalized.isna().any() or normalized.eq("").any():
        raise StockEODFamilyCorrupt(f"{timeframe} snapshot has invalid symbols")
    if normalized.duplicated().any():
        raise StockEODFamilyCorrupt(f"{timeframe} snapshot has duplicate symbols")
    return len(frame), int(normalized.nunique())


def _serialize_frame(frame: pd.DataFrame, timeframe: str) -> tuple[bytes, StockEODArtifact]:
    rows, symbols = _validate_frame(frame, timeframe)
    stream = BytesIO()
    frame.to_parquet(stream)
    data = stream.getvalue()
    return data, StockEODArtifact(
        filename=f"{timeframe}.parquet", sha256=_hash(data), byte_size=len(data),
        row_count=rows, symbol_count=symbols,
    )


def _create_immutable(path: Path, data: bytes, content_type: str) -> None:
    try:
        storage.create_bytes_if_absent(path, data, content_type=content_type)
    except storage.StoragePreconditionFailed:
        try:
            existing = storage.read_bytes(path)
        except Exception as exc:
            raise ImmutableStockEODFamilyConflict(f"immutable object conflict: {path}") from exc
        if existing != data:
            raise ImmutableStockEODFamilyConflict(f"immutable object conflict: {path}")


def _artifact_dict(item: StockEODArtifact) -> dict:
    return {
        "filename": item.filename, "sha256": item.sha256, "byte_size": item.byte_size,
        "row_count": item.row_count, "symbol_count": item.symbol_count,
    }


def _parse_manifest(data: bytes, run_id: str) -> StockEODFamilyManifest:
    try:
        raw = json.loads(data)
        if raw["schema_version"] != SCHEMA_VERSION or raw["producer"] != PRODUCER:
            raise ValueError("unsupported schema or producer")
        if validate_family_run_id(raw["family_run_id"]) != run_id:
            raise ValueError("manifest family identity mismatch")
        if set(raw["artifacts"]) != set(TIMEFRAMES):
            raise ValueError("manifest must contain exactly D/W/M")
        artifacts = {}
        for timeframe in TIMEFRAMES:
            item = raw["artifacts"][timeframe]
            if item["filename"] != f"{timeframe}.parquet":
                raise ValueError("non-canonical artifact filename")
            artifacts[timeframe] = StockEODArtifact(**item)
        return StockEODFamilyManifest(
            raw["schema_version"], raw["family_run_id"], raw["producer"],
            raw["staged_at_utc"], artifacts,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StockEODFamilyCorrupt(f"invalid manifest for {run_id}: {exc}") from exc


def _parse_pointer(data: bytes) -> StockEODFamilyPointer:
    try:
        raw = json.loads(data)
        run_id = validate_family_run_id(raw["family_run_id"])
        expected = str(_generation_relative(run_id) / "manifest.json")
        if raw["schema_version"] != SCHEMA_VERSION or raw["manifest_path"] != expected:
            raise ValueError("invalid pointer schema or manifest path")
        previous = raw["previous_family_run_id"]
        if previous is not None:
            validate_family_run_id(previous)
        return StockEODFamilyPointer(
            raw["schema_version"], run_id, raw["manifest_path"], raw["manifest_sha256"],
            raw["committed_at_utc"], previous,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, InvalidStockEODFamilyId) as exc:
        raise StockEODFamilyCorrupt(f"invalid current stock EOD family pointer: {exc}") from exc


def capture_current_pointer_state(data_root: Path = DATA) -> StockEODPointerState:
    current = storage.read_bytes_with_revision(current_pointer_path(data_root))
    if current.data is None:
        return StockEODPointerState(None, None)
    pointer = _parse_pointer(current.data)
    return StockEODPointerState(current.revision, pointer.family_run_id)


def _load_validated_generation(
    pointer: StockEODFamilyPointer, *, data_root: Path,
) -> StockEODSnapshotFamily:
    try:
        manifest_bytes = storage.read_bytes(manifest_path(pointer.family_run_id, data_root))
    except Exception as exc:
        raise StockEODFamilyCorrupt("committed stock EOD family manifest is missing") from exc
    if _hash(manifest_bytes) != pointer.manifest_sha256:
        raise StockEODFamilyCorrupt("stock EOD family manifest hash mismatch")
    manifest = _parse_manifest(manifest_bytes, pointer.family_run_id)
    frames = {}
    for timeframe in TIMEFRAMES:
        metadata = manifest.artifacts[timeframe]
        try:
            data = storage.read_bytes(artifact_path(pointer.family_run_id, timeframe, data_root))
        except Exception as exc:
            raise StockEODFamilyCorrupt(f"committed {timeframe} artifact is missing") from exc
        if len(data) != metadata.byte_size or _hash(data) != metadata.sha256:
            raise StockEODFamilyCorrupt(f"{timeframe} artifact integrity mismatch")
        try:
            frame = pd.read_parquet(BytesIO(data))
        except Exception as exc:
            raise StockEODFamilyCorrupt(f"{timeframe} artifact is unreadable") from exc
        rows, symbols = _validate_frame(frame, timeframe)
        if rows != metadata.row_count or symbols != metadata.symbol_count:
            raise StockEODFamilyCorrupt(f"{timeframe} artifact count mismatch")
        frames[timeframe] = frame
    return StockEODSnapshotFamily(pointer, manifest, frames["daily"], frames["weekly"], frames["monthly"])


def resolve_current_stock_eod_family(*, data_root: Path = DATA) -> StockEODSnapshotFamily:
    current = storage.read_bytes_with_revision(current_pointer_path(data_root))
    if current.data is None:
        raise StockEODFamilyNotPublished("stock EOD family has not been published")
    return _load_validated_generation(_parse_pointer(current.data), data_root=data_root)


def publish_stock_eod_family(
    family_run_id: str, daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame,
    *, expected_pointer_revision: str | None,
    expected_previous_family_run_id: str | None, data_root: Path = DATA,
) -> StockEODSnapshotFamily:
    validate_family_run_id(family_run_id)
    if expected_previous_family_run_id is not None:
        validate_family_run_id(expected_previous_family_run_id)
    if (expected_pointer_revision is None) != (expected_previous_family_run_id is None):
        raise StockEODFamilyCorrupt(
            "expected pointer revision and previous family identity must describe the same state"
        )
    serialized = {
        timeframe: _serialize_frame(frame, timeframe)
        for timeframe, frame in zip(TIMEFRAMES, (daily, weekly, monthly))
    }
    for timeframe in TIMEFRAMES:
        _create_immutable(
            artifact_path(family_run_id, timeframe, data_root), serialized[timeframe][0],
            "application/vnd.apache.parquet",
        )
    manifest_value = {
        "schema_version": SCHEMA_VERSION, "family_run_id": family_run_id,
        "producer": PRODUCER, "staged_at_utc": _run_staged_iso(family_run_id),
        "artifacts": {tf: _artifact_dict(serialized[tf][1]) for tf in TIMEFRAMES},
    }
    manifest_bytes = _json_bytes(manifest_value)
    manifest_object_path = manifest_path(family_run_id, data_root)
    _create_immutable(manifest_object_path, manifest_bytes, "application/json")
    staged_manifest_bytes = storage.read_bytes(manifest_object_path)
    manifest_hash = _hash(staged_manifest_bytes)
    staged_pointer = StockEODFamilyPointer(
        SCHEMA_VERSION, family_run_id, str(_generation_relative(family_run_id) / "manifest.json"),
        manifest_hash, _utc_iso(), expected_previous_family_run_id,
    )
    # Validate the exact stored bytes before making the generation authoritative.
    validated = _load_validated_generation(staged_pointer, data_root=data_root)
    pointer_bytes = _json_bytes({
        "schema_version": SCHEMA_VERSION, "family_run_id": family_run_id,
        "manifest_path": staged_pointer.manifest_path, "manifest_sha256": manifest_hash,
        "committed_at_utc": staged_pointer.committed_at_utc,
        "previous_family_run_id": expected_previous_family_run_id,
    })
    try:
        storage.compare_and_swap_bytes(
            current_pointer_path(data_root), expected_pointer_revision, pointer_bytes,
            content_type="application/json",
        )
    except storage.StoragePreconditionFailed as exc:
        raise ConcurrentStockEODPublication("stock EOD current pointer changed during publication") from exc
    return validated
