from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import s3fs


BUCKET = os.environ["S3_BUCKET_DATA"]
PREFIX = os.getenv("S3_PREFIX_DATA", "").strip("/")

symbol = os.getenv("FORENSIC_SYMBOL", "CAKE").strip().upper()
combo_name = os.getenv("FORENSIC_COMBO", "stocks_c_dwm_all").strip()
target_ts = pd.Timestamp(os.getenv("FORENSIC_TARGET_DATE", "2026-05-20"))
target_market_date = target_ts.date()

# Search a narrow execution-date window around the market date.
# The content predicate on lower_date remains authoritative.
search_dates = {
    (target_ts + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
    for offset in (-1, 0, 1, 2)
}

base = (
    f"{BUCKET}/{PREFIX}/combo_history/stocks/{combo_name}"
    if PREFIX
    else f"{BUCKET}/combo_history/stocks/{combo_name}"
)

fs = s3fs.S3FileSystem(
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    client_kwargs={
        "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    },
)

print(f"[FORENSICS] symbol: {symbol}")
print(f"[FORENSICS] combo: {combo_name}")
print(f"[FORENSICS] target market date: {target_market_date}")
print(f"[FORENSICS] scanning: s3://{base}")

paths = sorted(fs.glob(f"{base}/*.parquet"))

print(f"[FORENSICS] total history objects: {len(paths)}")

nearby = [
    p for p in paths
    if any(date_str in p for date_str in search_dates)
]

print(f"\n[FORENSICS] nearby execution objects: {len(nearby)}")
for p in nearby:
    print(" ", p)

if not nearby:
    raise RuntimeError(
        f"No combo-history objects found near {target_market_date} "
        f"for combo {combo_name}."
    )

print(
    f"\n[FORENSICS] searching contents for "
    f"{symbol} with lower_date = {target_market_date} ..."
)

exact_matches: list[tuple[str, pd.DataFrame]] = []
nearby_symbol_rows: list[dict[str, object]] = []

for p in nearby:
    with fs.open(p, "rb") as f:
        df = pd.read_parquet(f)

    if "symbol" not in df.columns:
        print(f"[FORENSICS] skipping object without symbol column: {p}")
        continue

    symbol_rows = df[
        df["symbol"].astype(str).str.upper() == symbol
    ].copy()

    if symbol_rows.empty:
        continue

    if "lower_date" in symbol_rows.columns:
        for _, row in symbol_rows.iterrows():
            nearby_symbol_rows.append(
                {
                    "source_object": p,
                    "symbol": symbol,
                    "lower_date": row.get("lower_date"),
                }
            )

        lower_dates = pd.to_datetime(
            symbol_rows["lower_date"], errors="coerce"
        ).dt.date

        target_rows = symbol_rows[
            lower_dates == target_market_date
        ].copy()

        if not target_rows.empty:
            target_rows.insert(0, "source_object", p)
            exact_matches.append((p, target_rows))

print(
    f"[FORENSICS] exact objects containing {symbol} on target "
    f"lower_date: {len(exact_matches)}"
)

out_dir = Path("forensic_artifacts")
out_dir.mkdir(parents=True, exist_ok=True)

safe_combo = combo_name.replace("/", "_")
target_str = target_ts.strftime("%Y-%m-%d")

if not exact_matches:
    print(
        f"\n[FORENSICS] No exact {symbol} lower_date="
        f"{target_market_date} match found."
    )

    if nearby_symbol_rows:
        nearby_df = pd.DataFrame(nearby_symbol_rows)
        print("\n[FORENSICS] nearby symbol/lower_date observations:")
        print(nearby_df.to_string(index=False))

        nearby_path = (
            out_dir
            / f"{symbol}_{safe_combo}_{target_str}_nearby_dates.csv"
        )
        nearby_df.to_csv(nearby_path, index=False)
        print(f"[FORENSICS] wrote {nearby_path}")

    raise RuntimeError(
        f"No exact combo-history row found for {symbol} "
        f"lower_date={target_market_date}."
    )

combined = pd.concat(
    [rows for _, rows in exact_matches],
    ignore_index=True,
)

combo_csv_path = (
    out_dir / f"{symbol}_{safe_combo}_{target_str}.csv"
)
combo_parquet_path = (
    out_dir / f"{symbol}_{safe_combo}_{target_str}.parquet"
)

combined.to_csv(combo_csv_path, index=False)
combined.to_parquet(combo_parquet_path, index=False)

print(f"[FORENSICS] wrote {combo_csv_path}")
print(f"[FORENSICS] wrote {combo_parquet_path}")

for p, rows in exact_matches:
    print(f"\n\n[FORENSICS] OBJECT: s3://{p}")

    for _, row in rows.iterrows():
        print(f"\n================ {symbol} ================")
        print(row.to_string())

print(
    f"\n[FORENSICS] total exact target rows exported: "
    f"{len(combined)}"
)
