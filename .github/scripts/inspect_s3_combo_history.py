from __future__ import annotations

import os

import pandas as pd
import s3fs


BUCKET = os.environ["S3_BUCKET_DATA"]
PREFIX = os.getenv("S3_PREFIX_DATA", "").strip("/")

combo_name = "stocks_c_dwm_all"
target_date = "2026-07-08"
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

matches = [
    p for p in paths
    if f"asof={target_date}" in p
]

print(f"[FORENSICS] total history objects: {len(paths)}")
print(f"[FORENSICS] {target_date} matches: {len(matches)}")

for p in matches:
    print(f"\n[FORENSICS] OBJECT: s3://{p}")

    with fs.open(p, "rb") as f:
        df = pd.read_parquet(f)

    if "symbol" not in df.columns:
        print("[WARN] no symbol column")
        continue

    rows = df[df["symbol"].astype(str).str.upper().isin(symbols)].copy()

    if rows.empty:
        print("[FORENSICS] Neither CAKE nor BBY present.")
        continue

    for sym in sorted(rows["symbol"].unique()):
        r = rows[rows["symbol"] == sym].iloc[0]

        print(f"\n===== {sym} =====")

        important = [
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

        for c in important:
            if c in r.index:
                print(f"{c}: {r[c]}")

        print("\n--- scoring / indicator fields ---")
        interesting_tokens = (
            "wyckoff",
            "exh_abs",
            "ma_trend",
            "macdv",
            "ttm_squeeze",
            "significant_volume",
            "spy_qqq",
            "etf",
        )

        for c in r.index:
            lc = str(c).lower()
            if any(token in lc for token in interesting_tokens):
                print(f"{c}: {r[c]}")
