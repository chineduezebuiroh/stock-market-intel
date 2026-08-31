from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import s3fs


BUCKET = os.environ["S3_BUCKET_DATA"]
PREFIX = os.getenv("S3_PREFIX_DATA", "").strip("/")

symbol = os.getenv("FORENSIC_SYMBOL", "CAKE").strip().upper()
target_date = pd.Timestamp(
    os.getenv("FORENSIC_TARGET_DATE", "2026-05-20")
)
rows_before = int(os.getenv("FORENSIC_ROWS_BEFORE", "180"))

key = (
    f"{PREFIX}/bars/stocks_daily/{symbol}.parquet"
    if PREFIX
    else f"bars/stocks_daily/{symbol}.parquet"
)

s3_path = f"{BUCKET}/{key}"

fs = s3fs.S3FileSystem(
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    client_kwargs={
        "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    },
)

print(f"[FORENSICS] symbol: {symbol}")
print(f"[FORENSICS] target date: {target_date.date()}")
print(f"[FORENSICS] rows before/through target: {rows_before}")
print(f"[FORENSICS] reading s3://{s3_path}")

with fs.open(s3_path, "rb") as f:
    df = pd.read_parquet(f)

print(f"[FORENSICS] source shape: {df.shape}")
print(f"[FORENSICS] columns: {list(df.columns)}")

# Normalize date/index without changing source values.
if isinstance(df.index, pd.DatetimeIndex):
    working = df.copy()
    working["_forensic_date"] = pd.to_datetime(
        working.index, errors="coerce"
    )
else:
    working = df.copy()

    date_col = None
    for candidate in ("date", "datetime", "timestamp"):
        if candidate in working.columns:
            date_col = candidate
            break

    if date_col is None:
        raise RuntimeError(
            "Could not determine date from index or columns."
        )

    working["_forensic_date"] = pd.to_datetime(
        working[date_col], errors="coerce"
    )

working = working[
    working["_forensic_date"].dt.normalize()
    <= target_date.normalize()
].sort_values("_forensic_date")

window = working.tail(rows_before).copy()

target_rows = window[
    window["_forensic_date"].dt.normalize()
    == target_date.normalize()
]

print(
    f"[FORENSICS] bounded rows through {target_date.date()}: "
    f"{len(window)}"
)
print(f"[FORENSICS] target-date rows: {len(target_rows)}")

if target_rows.empty:
    raise RuntimeError(
        f"{symbol} {target_date.date()} was not found in the "
        "surviving S3 rolling file."
    )

out_dir = Path("forensic_artifacts")
out_dir.mkdir(parents=True, exist_ok=True)

target_str = target_date.strftime("%Y-%m-%d")

csv_path = (
    out_dir / f"{symbol}_daily_through_{target_str}.csv"
)
parquet_path = (
    out_dir / f"{symbol}_daily_through_{target_str}.parquet"
)

window.to_csv(csv_path, index=False)
window.to_parquet(parquet_path, index=False)

print(f"[FORENSICS] wrote {csv_path}")
print(f"[FORENSICS] wrote {parquet_path}")

print("\n[FORENSICS] target row:")
print(target_rows.to_string(index=False))
