from __future__ import annotations

# core/storage.py

import os
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.paths import DATA

import s3fs
import boto3
from botocore.exceptions import ClientError

_DATA_BACKEND = os.getenv("DATA_BACKEND", "local").lower()


class StoragePreconditionFailed(RuntimeError):
    """An atomic object precondition did not match current storage state."""


@dataclass(frozen=True)
class BytesWithRevision:
    data: bytes | None
    revision: str | None


def _s3_bucket_prefix() -> tuple[str, str]:
    bucket = os.environ.get("S3_BUCKET_DATA")
    if not bucket:
        raise RuntimeError("S3_BUCKET_DATA is required when DATA_BACKEND='s3'")
    prefix = os.getenv("S3_PREFIX_DATA", "").strip("/")
    return bucket, prefix


def delete_s3_prefix(rel_prefix: str) -> None:
    """
    Delete everything under a prefix relative to DATA root.
    Example: rel_prefix="bars/futures_intraday_4h"
    """
    if _DATA_BACKEND != "s3":
        raise RuntimeError("delete_s3_prefix only valid for DATA_BACKEND='s3'")

    bucket, base_prefix = _s3_bucket_prefix()
    rel_prefix = rel_prefix.strip("/")

    full = f"{bucket}/{base_prefix}/{rel_prefix}" if base_prefix else f"{bucket}/{rel_prefix}"
    fs = _s3_fs()
    fs.rm(full, recursive=True)


def _ensure_path(path: str | Path) -> Path:
    """Normalize input into a Path object."""
    if isinstance(path, Path):
        return path
    return Path(path)


def _s3_storage_options() -> dict[str, Any]:
    """
    Storage options passed to pandas for s3.

    In most cases you can rely on standard AWS env vars:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION.
    So this can stay empty unless you want custom profiles/endpoints.
    """
    # Example if you ever need to customize:
    # return {"key": os.getenv("AWS_ACCESS_KEY_ID"),
    #         "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    #         "client_kwargs": {"region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1")}}
    return {}


def _rel_key_from_data(path: Path) -> str:
    """
    Compute the key relative to the DATA root.

    Example:
        DATA = /repo/data
        path = /repo/data/snapshot_stocks_daily.parquet
        -> 'snapshot_stocks_daily.parquet'

        path = /repo/data/combo_history/stocks/...
        -> 'combo_history/stocks/...'
    """
    try:
        rel = path.relative_to(DATA)
    except ValueError:
        # If path is not under DATA, just use the name (fallback).
        rel = path.name
    return str(rel).replace("\\", "/")  # normalize for S3


def _s3_uri_for_data_path(path: Path) -> str:
    """
    Map a local DATA-based path to an s3:// URI, using S3_BUCKET_DATA and S3_PREFIX_DATA.
    """
    bucket = os.environ.get("S3_BUCKET_DATA")
    if not bucket:
        raise RuntimeError("S3_BUCKET_DATA is required when DATA_BACKEND='s3'")

    prefix = os.getenv("S3_PREFIX_DATA", "").strip("/")
    rel_key = _rel_key_from_data(path)

    if prefix:
        key = f"{prefix}/{rel_key}"
    else:
        key = rel_key

    return f"s3://{bucket}/{key}"


def _s3_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={"region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1")},
    )


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _s3_bucket_key(path: Path) -> tuple[str, str]:
    bucket, prefix = _s3_bucket_prefix()
    rel = _rel_key_from_data(path)
    return bucket, f"{prefix}/{rel}" if prefix else rel


def _is_precondition_failure(exc: ClientError) -> bool:
    response = exc.response
    return response.get("ResponseMetadata", {}).get("HTTPStatusCode") in {409, 412} or str(
        response.get("Error", {}).get("Code", "")
    ) in {"ConditionalRequestConflict", "PreconditionFailed", "409", "412"}


def read_bytes_with_revision(path: str | Path) -> BytesWithRevision:
    """Read an object and its compare-and-swap revision, or return missing state."""
    p = _ensure_path(path)
    if _DATA_BACKEND == "local":
        try:
            data = p.read_bytes()
        except FileNotFoundError:
            return BytesWithRevision(None, None)
        return BytesWithRevision(data, hashlib.sha256(data).hexdigest())
    if _DATA_BACKEND == "s3":
        bucket, key = _s3_bucket_key(p)
        try:
            response = _s3_client().get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) in {"NoSuchKey", "NotFound", "404"}:
                return BytesWithRevision(None, None)
            raise
        return BytesWithRevision(response["Body"].read(), response["ETag"])
    raise RuntimeError("Unreachable: DATA_BACKEND validation failed")


def read_bytes(path: str | Path) -> bytes:
    result = read_bytes_with_revision(path)
    if result.data is None:
        raise FileNotFoundError(path)
    return result.data


def _local_locked_update(path: Path, expected_revision: str | None, data: bytes) -> str:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = path.read_bytes() if path.exists() else None
        revision = None if current is None else hashlib.sha256(current).hexdigest()
        if revision != expected_revision:
            raise StoragePreconditionFailed(
                f"object revision changed for {path}: expected={expected_revision!r} actual={revision!r}"
            )
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return hashlib.sha256(data).hexdigest()


def compare_and_swap_bytes(
    path: str | Path, expected_revision: str | None, data: bytes,
    *, content_type: str = "application/octet-stream",
) -> str:
    """Atomically create or replace an object iff its revision is unchanged."""
    p = _ensure_path(path)
    if _DATA_BACKEND == "local":
        return _local_locked_update(p, expected_revision, data)
    if _DATA_BACKEND == "s3":
        bucket, key = _s3_bucket_key(p)
        condition = {"IfNoneMatch": "*"} if expected_revision is None else {"IfMatch": expected_revision}
        try:
            response = _s3_client().put_object(
                Bucket=bucket, Key=key, Body=data, ContentType=content_type, **condition,
            )
        except ClientError as exc:
            if _is_precondition_failure(exc):
                raise StoragePreconditionFailed(f"object revision changed for {p}") from exc
            raise
        return response["ETag"]
    raise RuntimeError("Unreachable: DATA_BACKEND validation failed")


def create_bytes_if_absent(
    path: str | Path, data: bytes, *, content_type: str = "application/octet-stream",
) -> str:
    return compare_and_swap_bytes(path, None, data, content_type=content_type)


def load_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Load a Parquet file from either local disk or S3, depending on DATA_BACKEND.

    Usage:
        from core import storage
        df = storage.load_parquet(DATA / "snapshot_stocks_daily.parquet")
    """
    p = _ensure_path(path)

    if _DATA_BACKEND == "local":
        return pd.read_parquet(p, **kwargs)

    if _DATA_BACKEND == "s3":
        uri = _s3_uri_for_data_path(p)
        return pd.read_parquet(uri, storage_options=_s3_storage_options(), **kwargs)

    raise ValueError(f"Unsupported DATA_BACKEND: {_DATA_BACKEND!r}")


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """
    Save a Parquet file to either local disk or S3, depending on DATA_BACKEND.

    Usage:
        out_path = DATA / "snapshot_stocks_daily.parquet"
        storage.save_parquet(df, out_path)
    """
    p = _ensure_path(path)

    if _DATA_BACKEND == "local":
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, **kwargs)
        return

    if _DATA_BACKEND == "s3":
        uri = _s3_uri_for_data_path(p)
        df.to_parquet(uri, storage_options=_s3_storage_options(), **kwargs)
        return

    raise ValueError(f"Unsupported DATA_BACKEND: {_DATA_BACKEND!r}")

"""
def exists(path: str | Path) -> bool:
"""
"""
    Minimal existence check. For S3 this uses a cheap read attempt.
    Useful if you have any 'if file exists then...' logic.
"""
"""
    p = _ensure_path(path)
    if _DATA_BACKEND == "local":
        return p.exists()

    if _DATA_BACKEND == "s3":
        # Very simple check: try reading just the metadata / fail fast.
        # You can optimize later with boto3 if needed.
        try:
            uri = _s3_uri_for_data_path(p)
            # small hack: read only the schema (pyarrow), but pandas doesn't expose that cleanly.
            # For now, we just try to read and catch failures.
            _ = pd.read_parquet(uri, storage_options=_s3_storage_options(), columns=[])
            return True
        except Exception:
            return False

    raise ValueError(f"Unsupported DATA_BACKEND: {_DATA_BACKEND!r}")
"""


def exists(path: str | Path) -> bool:
    p = _ensure_path(path)
    if _DATA_BACKEND == "local":
        return p.exists()

    if _DATA_BACKEND == "s3":
        uri = _s3_uri_for_data_path(p)          # s3://bucket/prefix/...
        fs = _s3_fs()
        # s3fs expects bucket/key style without scheme:
        key = uri.replace("s3://", "", 1)
        return fs.exists(key)

    #raise ValueError(f"Unsupported DATA_BACKEND: {_DATA_BACKEND!r}")
    raise RuntimeError("Unreachable: DATA_BACKEND validation failed")
