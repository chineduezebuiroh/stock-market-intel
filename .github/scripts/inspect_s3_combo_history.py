from __future__ import annotations

import os
import pandas as pd
import s3fs


BUCKET = os.environ["S3_BUCKET_DATA"]
PREFIX = os.getenv("S3_PREFIX_DATA", "").strip("/")

combo_name = "stocks_c_dwm_all"
target_market_date = pd.Timestamp("2026-07-08").date()
symbols = {"CAKE", "BBY"}

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

print(f"[FORENSICS] scanning: s3://{base}")

paths = sorted(fs.glob(f"{base}/*.parquet"))

print(f"[FORENSICS] total history objects: {len(paths)}")

# First show filenames around the market date so we understand execution timing.
nearby = [
    p for p in paths
    if any(
        d in p
        for d in (
            "2026-07-07",
            "2026-07-08",
            "2026-07-09",
            "2026-07-10",
        )
    )
]

print(f"\n[FORENSICS] nearby execution objects: {len(nearby)}")
for p in nearby:
    print(" ", p)

print("\n[FORENSICS] searching contents for lower_date = 2026-07-08 ...")

matches = []

for p in nearby:
    with fs.open(p, "rb") as f:
        df = pd.read_parquet(f)

    if "lower_date" not in df.columns:
        continue

    lower_dates = pd.to_datetime(df["lower_date"], errors="coerce").dt.date

    if (lower_dates == target_market_date).any():
        matches.append((p, df))

print(f"[FORENSICS] objects containing target lower_date: {len(matches)}")

for p, df in matches:
    print(f"\n\n[FORENSICS] OBJECT: s3://{p}")

    rows = df[
        df["symbol"]
        .astype(str)
        .str.upper()
        .isin(symbols)
    ].copy()

    if rows.empty:
        print("[FORENSICS] CAKE/BBY not present in this object.")
        continue

    for _, r in rows.iterrows():
        sym = str(r["symbol"]).upper()

        print(f"\n================ {sym} ================")

        basic = [
            "lower_date",
            "middle_date",
            "upper_date",

            "lower_open",
            "lower_high",
            "lower_low",
            "lower_close",

            "middle_open",
            "middle_high",
            "middle_low",
            "middle_close",

            "upper_open",
            "upper_high",
            "upper_low",
            "upper_close",

            "mtf_long_score",
            "mtf_short_score",
            "signal",
            "signal_side",

            "etf_symbol_primary",
            "etf_symbol_secondary",
        ]

        for c in basic:
            if c in r.index:
                print(f"{c}: {r[c]}")

        print("\n--- indicator / scoring fields ---")

        interesting_tokens = (
            "wyckoff",
            "exh_abs",
            "sig_vol",
            "ma_trend",
            "macdv",
            "ttm_squeeze",
            "spy_qqq",
            "etf",
        )

        for c in r.index:
            lc = str(c).lower()
            if any(token in lc for token in interesting_tokens):
                print(f"{c}: {r[c]}")
