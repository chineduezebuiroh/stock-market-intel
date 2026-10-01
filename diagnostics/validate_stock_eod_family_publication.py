"""Read-only validation of the persisted Stock EOD publication contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys

import pandas as pd

from core import storage
from core.paths import DATA
from core import stock_eod_family as family


COMBOS = ("stocks_c_dwm_shortlist", "stocks_c_dwm_all")
LEGACY_MIRRORS = {
    timeframe: f"snapshot_stocks_{timeframe}.parquet" for timeframe in family.TIMEFRAMES
}


class DiagnosticFailure(RuntimeError):
    """A persisted publication invariant did not hold."""


@dataclass(frozen=True)
class ValidationReport:
    family_run_id: str
    previous_family_run_id: str | None
    pointer_revision: str | None


def _fail(invariant: str, expected: object, observed: object) -> None:
    raise DiagnosticFailure(f"{invariant}: expected={expected!r} observed={observed!r}")


def _read_required_bytes(path: Path, invariant: str) -> bytes:
    try:
        return storage.read_bytes(path)
    except FileNotFoundError:
        _fail(invariant, "object exists", "missing")
    except Exception as exc:
        raise DiagnosticFailure(f"{invariant}: unable to read {path}: {exc}") from exc


def _validate_generation(
    run_id: str,
    *,
    data_root: Path,
    expected_manifest_sha256: str | None = None,
    label: str = "current",
) -> tuple[family.StockEODFamilyManifest, dict[str, pd.DataFrame]]:
    manifest_object = family.manifest_path(run_id, data_root)
    manifest_bytes = _read_required_bytes(manifest_object, f"{label} manifest exists")
    observed_manifest_hash = sha256(manifest_bytes).hexdigest()
    if expected_manifest_sha256 is not None and observed_manifest_hash != expected_manifest_sha256:
        _fail(f"{label} manifest byte hash", expected_manifest_sha256, observed_manifest_hash)
    try:
        manifest = family._parse_manifest(manifest_bytes, run_id)
    except family.StockEODFamilyError as exc:
        raise DiagnosticFailure(f"{label} manifest schema/family: {exc}") from exc

    frames: dict[str, pd.DataFrame] = {}
    for timeframe in family.TIMEFRAMES:
        metadata = manifest.artifacts[timeframe]
        path = family.artifact_path(run_id, timeframe, data_root)
        data = _read_required_bytes(path, f"{label} {timeframe} immutable artifact exists")
        if len(data) != metadata.byte_size:
            _fail(f"{label} {timeframe} artifact byte_size", metadata.byte_size, len(data))
        digest = sha256(data).hexdigest()
        if digest != metadata.sha256:
            _fail(f"{label} {timeframe} artifact byte hash", metadata.sha256, digest)
        try:
            frame = pd.read_parquet(BytesIO(data))
        except Exception as exc:
            raise DiagnosticFailure(f"{label} {timeframe} artifact parquet decode: {exc}") from exc
        try:
            rows, symbols = family._validate_frame(frame, timeframe)
        except family.StockEODFamilyError as exc:
            raise DiagnosticFailure(f"{label} {timeframe} artifact structure: {exc}") from exc
        if rows != metadata.row_count:
            _fail(f"{label} {timeframe} artifact row_count", metadata.row_count, rows)
        if symbols != metadata.symbol_count:
            _fail(f"{label} {timeframe} artifact symbol_count", metadata.symbol_count, symbols)
        frames[timeframe] = frame
        print(
            f"[PASS] {timeframe} artifact path={path} rows={rows} symbols={symbols} "
            f"sha256={digest[:12]}..."
        )
    return manifest, frames


def _validate_combo(name: str, pointer: family.StockEODFamilyPointer, data_root: Path) -> None:
    path = data_root / f"combo_{name}.parquet"
    try:
        frame = storage.load_parquet(path)
    except FileNotFoundError:
        _fail(f"{name} combo exists", "object exists", "missing")
    except Exception as exc:
        raise DiagnosticFailure(f"{name} combo read: {exc}") from exc
    if frame.empty:
        _fail(f"{name} combo nonempty", "> 0 rows", 0)

    run_column = "stock_eod_family_run_id"
    committed_column = "stock_eod_family_committed_at_utc"
    for column in (run_column, committed_column):
        if column not in frame.columns:
            _fail(f"{name} combo lineage column", column, "missing")
        if frame[column].isna().any():
            _fail(f"{name} combo {column} non-null", "no nulls", "null value present")
        if frame[column].astype("string").str.strip().eq("").any():
            _fail(f"{name} combo {column} populated", "nonblank values", "blank value present")

    run_ids = sorted(str(value) for value in frame[run_column].unique())
    committed_values = sorted(str(value) for value in frame[committed_column].unique())
    if len(run_ids) != 1:
        _fail(f"{name} combo single family lineage", 1, run_ids)
    if run_ids[0] != pointer.family_run_id:
        _fail(f"{name} combo current family lineage", pointer.family_run_id, run_ids[0])
    if committed_values != [pointer.committed_at_utc]:
        _fail(
            f"{name} combo committed-at lineage",
            [pointer.committed_at_utc],
            committed_values,
        )
    print(
        f"[PASS] {name} combo lineage path={path} rows={len(frame)} "
        f"family_run_ids={run_ids} committed_at_utc={committed_values}"
    )


def validate_publication_contract(
    *,
    data_root: Path = DATA,
    expected_family_run_id: str | None = None,
    expected_previous_family_run_id: str | None = None,
) -> ValidationReport:
    """Validate production objects exclusively through read APIs."""
    pointer_path = family.current_pointer_path(data_root)
    try:
        current = storage.read_bytes_with_revision(pointer_path)
    except Exception as exc:
        raise DiagnosticFailure(f"current pointer read: {exc}") from exc
    if current.data is None:
        _fail("current pointer exists", "object exists", "missing")
    try:
        pointer = family._parse_pointer(current.data)
    except family.StockEODFamilyError as exc:
        raise DiagnosticFailure(f"current pointer schema: {exc}") from exc

    if expected_family_run_id is not None and pointer.family_run_id != expected_family_run_id:
        _fail("expected current family", expected_family_run_id, pointer.family_run_id)
    if (
        expected_previous_family_run_id is not None
        and pointer.previous_family_run_id != expected_previous_family_run_id
    ):
        _fail(
            "expected previous family",
            expected_previous_family_run_id,
            pointer.previous_family_run_id,
        )
    print(
        f"[PASS] pointer path={pointer_path} schema_version={pointer.schema_version} "
        f"family_run_id={pointer.family_run_id} manifest_path={pointer.manifest_path} "
        f"manifest_sha256={pointer.manifest_sha256[:12]}... "
        f"committed_at_utc={pointer.committed_at_utc} "
        f"previous_family_run_id={pointer.previous_family_run_id} revision={current.revision}"
    )

    manifest, _ = _validate_generation(
        pointer.family_run_id,
        data_root=data_root,
        expected_manifest_sha256=pointer.manifest_sha256,
    )
    if manifest.family_run_id != pointer.family_run_id:
        _fail("pointer -> manifest family", pointer.family_run_id, manifest.family_run_id)
    expected_manifest_path = str(
        family.manifest_path(pointer.family_run_id, data_root).relative_to(data_root)
    )
    if pointer.manifest_path != expected_manifest_path:
        _fail("pointer -> manifest path", expected_manifest_path, pointer.manifest_path)
    print(
        f"[PASS] manifest path={pointer.manifest_path} schema_version={manifest.schema_version} "
        f"family_run_id={manifest.family_run_id} sha256={pointer.manifest_sha256[:12]}..."
    )

    for name in COMBOS:
        _validate_combo(name, pointer, data_root)

    previous = pointer.previous_family_run_id
    if previous is None:
        _fail("previous generation identity", "non-null family_run_id", None)
    _validate_generation(previous, data_root=data_root, label="previous")
    print(f"[PASS] previous generation retained family_run_id={previous}")

    legacy_counts = {}
    for timeframe, filename in LEGACY_MIRRORS.items():
        path = data_root / filename
        try:
            frame = storage.load_parquet(path)
        except FileNotFoundError:
            _fail(f"legacy {timeframe} mirror exists", "object exists", "missing")
        except Exception as exc:
            raise DiagnosticFailure(f"legacy {timeframe} mirror read: {exc}") from exc
        legacy_counts[timeframe] = len(frame)
    print(
        "[PASS] legacy mirrors present compatibility_only=true "
        + " ".join(f"{name}_rows={count}" for name, count in legacy_counts.items())
    )
    print(
        "[PASS] publication contract validated read-only "
        "authoritative=current.json->immutable_generation compatibility_only=legacy_snapshot_mirrors"
    )
    return ValidationReport(pointer.family_run_id, previous, current.revision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-family-run-id")
    parser.add_argument("--expected-previous-family-run-id")
    args = parser.parse_args(argv)
    try:
        validate_publication_contract(
            expected_family_run_id=args.expected_family_run_id or None,
            expected_previous_family_run_id=args.expected_previous_family_run_id or None,
        )
    except DiagnosticFailure as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
