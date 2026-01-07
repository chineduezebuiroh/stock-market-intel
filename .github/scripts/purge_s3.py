from __future__ import annotations

# scripts/purge_s3.py

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import boto3


@dataclass(frozen=True)
class PurgeSpec:
    name: str
    paths: tuple[str, ...]  # relative to S3_PREFIX_DATA, may end with "/" for prefix delete


SPECS: dict[str, PurgeSpec] = {
    "futures_poisoned_4hdw": PurgeSpec(
        name="futures_poisoned_4hdw",
        paths=(
            "bars/futures_intraday_1h/",
            "bars/futures_intraday_4h/",
            "snapshot_futures_intraday_1h.parquet",
            "snapshot_futures_intraday_4h.parquet",
            "combo_futures_2_4hdw_shortlist.parquet",
            "combo_history/futures/futures_2_4hdw_shortlist/",
        ),
    ),
    "futures_all_intraday": PurgeSpec(
        name="futures_all_intraday",
        paths=(
            "bars/futures_intraday_1h/",
            "bars/futures_intraday_4h/",
            "snapshot_futures_intraday_1h.parquet",
            "snapshot_futures_intraday_4h.parquet",
        ),
    ),
}


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _normalize_rel(rel: str) -> str:
    rel = rel.strip()
    rel = rel.lstrip("/")  # never allow absolute-ish
    if rel in ("", ".", ".."):
        raise ValueError("Refusing to purge empty/unsafe path.")
    return rel


def _full_key(prefix: str, rel: str) -> str:
    prefix = prefix.strip().strip("/")
    rel = _normalize_rel(rel)
    return f"{prefix}/{rel}" if prefix else rel


def _iter_keys_for_path(
    s3,
    bucket: str,
    key_or_prefix: str,
) -> Iterable[str]:
    """
    If key_or_prefix endswith '/', treat as prefix delete.
    Else treat as single object delete.
    """
    if key_or_prefix.endswith("/"):
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key_or_prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]
    else:
        yield key_or_prefix


def purge(*, target: str, dry_run: bool, custom_paths: list[str] | None = None) -> int:
    if target == "custom_list":
        if not custom_paths:
            raise RuntimeError("custom_list selected but no custom paths provided.")
        paths = tuple(_normalize_rel(p) for p in custom_paths if p.strip())
    else:
        spec = SPECS.get(target)
        if not spec:
            raise RuntimeError(f"Unknown target {target!r}. Allowed: {list(SPECS)} or custom_list.")
        paths = spec.paths

    backend = os.getenv("DATA_BACKEND", "local").lower()
    if backend != "s3":
        raise RuntimeError(f"Refusing to purge unless DATA_BACKEND=s3 (got {backend!r})")

    bucket = _require_env("S3_BUCKET_DATA")
    prefix = os.getenv("S3_PREFIX_DATA", "").strip("/")

    s3 = boto3.client("s3")

    # Expand to actual keys (for prefix deletes)
    expanded: list[str] = []
    for rel in paths:
        key = _full_key(prefix, rel)
        expanded.extend(list(_iter_keys_for_path(s3, bucket, key)))

    # Dedupe but keep stable order
    seen = set()
    expanded_unique = []
    for k in expanded:
        if k not in seen:
            seen.add(k)
            expanded_unique.append(k)

    print(f"[PURGE] bucket={bucket} prefix={prefix or '(root)'} target={target} dry_run={dry_run}")
    print(f"[PURGE] requested paths ({len(paths)}):")
    for p in paths:
        print(f"  - {p}")

    print(f"[PURGE] resolved object keys ({len(expanded_unique)}):")
    for k in expanded_unique[:5000]:
        print(f"  - s3://{bucket}/{k}")
    if len(expanded_unique) > 5000:
        print(f"  ... truncated, total={len(expanded_unique)}")

    if dry_run:
        print("[PURGE] dry_run=True → no deletions executed.")
        return 0

    # Delete in batches of 1000
    CHUNK = 1000
    for i in range(0, len(expanded_unique), CHUNK):
        chunk = expanded_unique[i : i + CHUNK]
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": False},
        )
        deleted = resp.get("Deleted", [])
        errors = resp.get("Errors", [])
        print(f"[PURGE] batch {i//CHUNK + 1}: deleted={len(deleted)} errors={len(errors)}")
        if errors:
            for e in errors[:50]:
                print(f"[PURGE][ERR] {e}")
            raise RuntimeError("Delete errors encountered; aborting.")

    print("[PURGE] completed.")
    return 0


def main(argv: list[str]) -> int:
    # args: target dry_run [custom_paths_file]
    if len(argv) < 3:
        print("Usage: python scripts/purge_s3.py <target> <dry_run:true|false> [custom_paths_file]")
        print(f"Targets: {list(SPECS.keys())} + custom_list")
        return 2

    target = argv[1].strip()
    dry_run = argv[2].strip().lower() in ("1", "true", "yes", "y")

    custom_paths = None
    if target == "custom_list":
        if len(argv) < 4:
            raise RuntimeError("custom_list requires custom_paths_file")
        path_file = argv[3]
        with open(path_file, "r", encoding="utf-8") as f:
            custom_paths = [line.rstrip("\n") for line in f]

    return purge(target=target, dry_run=dry_run, custom_paths=custom_paths)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
