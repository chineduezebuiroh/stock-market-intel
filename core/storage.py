from __future__ import annotations

# core/storage.py

import os
from pathlib import Path
from typing import Any

import pandas as pd

from core.paths import DATA

import s3fs

_DATA_BACKEND = os.getenv("DATA_BACKEND", "local").lower()


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
