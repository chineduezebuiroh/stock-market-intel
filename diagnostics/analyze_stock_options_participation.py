#!/usr/bin/env python3
"""Read-only inventory and score validation for stock-options D/W/M history.

The program deliberately bypasses ``core.storage``.  Production input is read
with the S3 ListObjectsV2 and GetObject APIs, and every output is written to a
caller-selected local directory.  It never calls an S3 mutation API.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import boto3
import numpy as np
import pandas as pd

MODERN_REQUIRED_FIELDS = [
    "symbol",
    "lower_date",
    "middle_date",
    "upper_date",
    "upper_wyckoff_stage",
    "upper_exh_abs_pa_prior_bar",
    "middle_wyckoff_stage",
    "middle_exh_abs_pa_prior_bar",
    "lower_ma_trend_bullish",
    "lower_ma_trend_bearish",
    "lower_exh_abs_pa_current_bar",
    "lower_macdv_core_bull",
    "lower_macdv_core_bear",
    "lower_ttm_squeeze_pro",
    "lower_sig_vol_current_bar",
    "lower_sig_vol_prior_bar",
    "lower_spy_qqq_vol_ma_ratio",
    "middle_sig_vol_current_bar",
    "middle_sig_vol_prior_bar",
    "middle_spy_qqq_vol_ma_ratio",
    "mtf_long_score",
    "mtf_short_score",
    "signal",
    "signal_side",
    "lower_open",
    "lower_high",
    "lower_low",
    "lower_close",
    "lower_volume",
]

SCORING_NUMERIC_FIELDS = [
    "upper_wyckoff_stage",
    "upper_exh_abs_pa_prior_bar",
    "middle_wyckoff_stage",
    "middle_exh_abs_pa_prior_bar",
    "lower_ma_trend_bullish",
    "lower_ma_trend_bearish",
    "lower_exh_abs_pa_current_bar",
    "lower_macdv_core_bull",
    "lower_macdv_core_bear",
    "lower_ttm_squeeze_pro",
    "lower_sig_vol_current_bar",
    "lower_sig_vol_prior_bar",
    "lower_spy_qqq_vol_ma_ratio",
    "middle_sig_vol_current_bar",
    "middle_sig_vol_prior_bar",
    "middle_spy_qqq_vol_ma_ratio",
    "mtf_long_score",
    "mtf_short_score",
]

LOWER_OHLCV = [
    "lower_open",
    "lower_high",
    "lower_low",
    "lower_close",
    "lower_volume",
]

AUDIT_COMPARE_FIELDS = LOWER_OHLCV + [
    "upper_wyckoff_stage",
    "upper_exh_abs_pa_prior_bar",
    "middle_wyckoff_stage",
    "middle_exh_abs_pa_prior_bar",
    "lower_ma_trend_bullish",
    "lower_ma_trend_bearish",
    "lower_exh_abs_pa_current_bar",
    "lower_macdv_core_bull",
    "lower_macdv_core_bear",
    "lower_ttm_squeeze_pro",
    "lower_sig_vol_current_bar",
    "lower_sig_vol_prior_bar",
    "lower_spy_qqq_vol_ma_ratio",
    "middle_sig_vol_current_bar",
    "middle_sig_vol_prior_bar",
    "middle_spy_qqq_vol_ma_ratio",
]

DERIVED_BOOL_FIELDS = [
    "regime_long_pass",
    "ma_long_pass",
    "price_action_long_pass",
    "momentum_long_pass",
    "pre_participation_long",
    "regime_short_pass",
    "ma_short_pass",
    "price_action_short_pass",
    "momentum_short_pass",
    "pre_participation_short",
    "lower_route_pass",
    "middle_route_pass",
    "participation_pass",
]

SUPPORTED_OUTPUT_FIELDS = (
    [
        "source_s3_key",
        "artifact_execution_utc",
        "logic_era",
        "symbol",
        "lower_date",
        "middle_date",
        "upper_date",
    ]
    + DERIVED_BOOL_FIELDS
    + [
        "lower_sig_vol_current_bar",
        "lower_sig_vol_prior_bar",
        "lower_spy_qqq_vol_ma_ratio",
        "middle_sig_vol_current_bar",
        "middle_sig_vol_prior_bar",
        "middle_spy_qqq_vol_ma_ratio",
        "upper_wyckoff_stage",
        "reconstructed_long_score",
        "mtf_long_score",
        "reconstructed_short_score",
        "mtf_short_score",
        "signal",
        "signal_side",
    ]
    + LOWER_OHLCV
)

EXCEPTION_FIELDS = (
    [
        "source_s3_key",
        "artifact_execution_utc",
        "symbol",
        "lower_date",
    ]
    + DERIVED_BOOL_FIELDS
    + [
        "reconstructed_long_score",
        "mtf_long_score",
        "reconstructed_short_score",
        "mtf_short_score",
    ]
    + SCORING_NUMERIC_FIELDS
)

FIVE_COMPONENT_ERA = "MODERN_SUPPORTED_FIVE_COMPONENT"
PRE_PARTICIPATION_ERA = "MODERN_PRE_PARTICIPATION_SCORE"

KNOWN_CASES = [
    {
        "case_id": "BBY_2026-07-08",
        "symbol": "BBY",
        "lower_date": "2026-07-08",
        "expected": {
            "pre_participation_long": True,
            "participation_pass": False,
        },
        "approx": {
            "lower_sig_vol_current_bar": (1.0, 0.0),
            "lower_spy_qqq_vol_ma_ratio": (0.0766, 0.002),
            "middle_sig_vol_current_bar": (1.0, 0.0),
            "middle_spy_qqq_vol_ma_ratio": (0.0804, 0.002),
        },
    },
    {
        "case_id": "CAKE_2026-05-20",
        "symbol": "CAKE",
        "lower_date": "2026-05-20",
        "expected": {"pre_participation_long": False},
        "approx": {},
    },
    {
        "case_id": "CAKE_2026-06-02",
        "symbol": "CAKE",
        "lower_date": "2026-06-02",
        "expected": {
            "price_action_long_pass": False,
            "pre_participation_long": False,
        },
        "approx": {},
    },
    {
        "case_id": "CAKE_2026-07-24",
        "symbol": "CAKE",
        "lower_date": "2026-07-24",
        "expected": {
            "pre_participation_long": True,
            "participation_pass": False,
        },
        "approx": {
            "lower_sig_vol_current_bar": (2.0, 0.0),
            "lower_spy_qqq_vol_ma_ratio": (0.03738, 0.002),
            "middle_sig_vol_current_bar": (2.0, 0.0),
            "middle_spy_qqq_vol_ma_ratio": (0.02482, 0.002),
        },
    },
]

EXECUTION_RE = re.compile(
    r"_asof=(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.parquet$"
)


@dataclass(frozen=True)
class SourceObject:
    key: str
    execution_utc: pd.Timestamp
    local_path: Path | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="stock-intel-data-prod")
    parser.add_argument("--prefix", default="data")
    parser.add_argument("--combo", default="stocks_c_dwm_all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dedupe",
        choices=["latest-same-market-date"],
        default="latest-same-market-date",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Require complete modern fields and exact scores before support classification.",
    )
    parser.add_argument(
        "--phase", choices=["inventory", "validate"], default="inventory"
    )
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument(
        "--include-unsupported-era-in-inventory",
        action="store_true",
        help="Include full column manifests for quarantined schemas in schema_eras.csv.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete and recreate the local output directory if it already exists.",
    )
    parser.add_argument(
        "--local-history-dir",
        help="Read local parquet history instead of S3 (tests/development only).",
    )
    return parser.parse_args(argv)


def parse_date(value: str | None, flag: str) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas exception varies
        raise ValueError(f"{flag} must be a valid YYYY-MM-DD date: {value!r}") from exc
    if parsed.time() != datetime.min.time():
        raise ValueError(f"{flag} must not include a time: {value!r}")
    return parsed.normalize().tz_localize(None)


def prepare_output_dir(path_text: str, overwrite: bool) -> Path:
    if "://" in path_text:
        raise ValueError("--output-dir must be a local filesystem path, not a URI")
    path = Path(path_text).expanduser().resolve()
    production_data = (Path(__file__).resolve().parents[1] / "data").resolve()
    if path == production_data or production_data in path.parents:
        raise ValueError("refusing to write diagnostic output under production data/")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {path}; use --overwrite-output explicitly"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def execution_from_name(name: str) -> pd.Timestamp:
    match = EXECUTION_RE.search(name)
    if not match:
        raise ValueError(f"cannot parse execution timestamp from {name!r}")
    stamp = datetime.strptime(match.group("stamp"), "%Y-%m-%dT%H-%M-%S")
    return pd.Timestamp(stamp, tz="UTC")


def combo_prefix(prefix: str, combo: str) -> str:
    parts = [prefix.strip("/"), "combo_history", "stocks", combo]
    return "/".join(part for part in parts if part) + "/"


def list_s3_objects(bucket: str, key_prefix: str) -> tuple[Any, list[SourceObject]]:
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    objects: list[SourceObject] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if not key.endswith(".parquet"):
                continue
            objects.append(
                SourceObject(key=key, execution_utc=execution_from_name(key))
            )
    objects.sort(key=lambda obj: (obj.execution_utc, obj.key))
    return client, objects


def list_local_objects(directory: Path) -> tuple[None, list[SourceObject]]:
    objects = [
        SourceObject(
            key=str(path.resolve()),
            execution_utc=execution_from_name(path.name),
            local_path=path,
        )
        for path in directory.glob("*.parquet")
    ]
    objects.sort(key=lambda obj: (obj.execution_utc, obj.key))
    return None, objects


def read_object(client: Any, bucket: str, source: SourceObject) -> pd.DataFrame:
    if source.local_path is not None:
        return pd.read_parquet(source.local_path)
    response = client.get_object(Bucket=bucket, Key=source.key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


def schema_signature(columns: Iterable[Any]) -> str:
    normalized = "\n".join(sorted(str(column) for column in columns))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def provisional_era(columns: Iterable[Any]) -> tuple[str, list[str]]:
    column_set = {str(column) for column in columns}
    missing = sorted(set(MODERN_REQUIRED_FIELDS) - column_set)
    if not missing:
        return "MODERN_SUPPORTED_CANDIDATE", []
    has_combined = "lower_macdv_core" in column_set
    lacks_split = not {
        "lower_macdv_core_bull",
        "lower_macdv_core_bear",
    }.issubset(column_set)
    if has_combined and lacks_split:
        return "LEGACY_COMBINED_MACDV", missing
    return "UNKNOWN_OR_MIXED", missing


def normalize_dates(series: pd.Series) -> tuple[pd.Series, int]:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    malformed = int((series.notna() & parsed.isna()).sum())
    return parsed.dt.tz_convert(None).dt.normalize(), malformed


def numeric_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    numbers = pd.DataFrame(index=df.index)
    malformed: dict[str, int] = {}
    for field in SCORING_NUMERIC_FIELDS:
        converted = pd.to_numeric(df[field], errors="coerce")
        malformed[field] = int((df[field].notna() & converted.isna()).sum())
        numbers[field] = converted
    return numbers, malformed


def reconstruct_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    numbers, malformed = numeric_frame(df)
    result = df.copy()

    u = numbers["upper_wyckoff_stage"]
    upper_available = u.notna()
    ue = numbers["upper_exh_abs_pa_prior_bar"]
    mw = numbers["middle_wyckoff_stage"]
    me = numbers["middle_exh_abs_pa_prior_bar"]

    result["regime_long_pass"] = (upper_available & ((u > 0) | (ue > 0))) | (
        ~upper_available & ((mw > 0) | (me > 0))
    )
    result["regime_short_pass"] = (upper_available & ((u < 0) | (ue < 0))) | (
        ~upper_available & ((mw < 0) | (me < 0))
    )

    result["ma_long_pass"] = numbers["lower_ma_trend_bullish"] > 0
    result["ma_short_pass"] = numbers["lower_ma_trend_bearish"] < 0

    price_action = numbers["lower_exh_abs_pa_current_bar"]
    result["price_action_long_pass"] = price_action.isin([1.0, 2.0])
    result["price_action_short_pass"] = price_action.isin([-1.0, -2.0])

    bull = numbers["lower_macdv_core_bull"]
    bear = numbers["lower_macdv_core_bear"]
    squeeze = numbers["lower_ttm_squeeze_pro"]
    result["momentum_long_pass"] = (bull == 2) | (
        (bull == 1) & squeeze.notna() & (squeeze >= 0)
    )
    result["momentum_short_pass"] = (bear == -2) | (
        (bear == -1) & squeeze.notna() & (squeeze <= 0)
    )

    result["pre_participation_long"] = result[
        [
            "regime_long_pass",
            "ma_long_pass",
            "price_action_long_pass",
            "momentum_long_pass",
        ]
    ].all(axis=1)
    result["pre_participation_short"] = result[
        [
            "regime_short_pass",
            "ma_short_pass",
            "price_action_short_pass",
            "momentum_short_pass",
        ]
    ].all(axis=1)

    strong_threshold = pd.Series(np.where(upper_available, 0.05, 0.10), index=df.index)
    ls = numbers["lower_sig_vol_current_bar"]
    lr = numbers["lower_spy_qqq_vol_ma_ratio"]
    ms = numbers["middle_sig_vol_current_bar"]
    mr = numbers["middle_spy_qqq_vol_ma_ratio"]
    result["lower_route_pass"] = ((ls == 2) & (lr > strong_threshold)) | (
        (ls == 1) & (lr > 0.25)
    )
    result["middle_route_pass"] = ((ms == 2) & (mr > strong_threshold)) | (
        (ms == 1) & (mr > 0.25)
    )
    result["participation_pass"] = (
        result["lower_route_pass"] | result["middle_route_pass"]
    )

    long_components = [
        "regime_long_pass",
        "ma_long_pass",
        "price_action_long_pass",
        "momentum_long_pass",
        "participation_pass",
    ]
    short_components = [
        "regime_short_pass",
        "ma_short_pass",
        "price_action_short_pass",
        "momentum_short_pass",
        "participation_pass",
    ]
    result["reconstructed_long_score"] = result[long_components].astype(int).sum(axis=1)
    result["reconstructed_short_score"] = (
        result[short_components].astype(int).sum(axis=1)
    )
    result["long_score_match"] = (
        result["reconstructed_long_score"] == numbers["mtf_long_score"]
    )
    result["short_score_match"] = (
        result["reconstructed_short_score"] == numbers["mtf_short_score"]
    )
    result["either_score_mismatch"] = ~(
        result["long_score_match"] & result["short_score_match"]
    )
    return result, malformed


def score_pattern_summary(reconstructed: pd.DataFrame) -> dict[str, Any]:
    """Describe score deltas and recognize the pre-participation scoring contract."""
    checked = reconstructed.copy()
    checked["long_delta"] = checked["reconstructed_long_score"] - pd.to_numeric(
        checked["mtf_long_score"], errors="coerce"
    )
    checked["short_delta"] = checked["reconstructed_short_score"] - pd.to_numeric(
        checked["mtf_short_score"], errors="coerce"
    )
    mismatches = checked[checked["either_score_mismatch"]]
    nonparticipation = checked[~checked["participation_pass"]]
    nonparticipation_mismatches = nonparticipation[
        nonparticipation["either_score_mismatch"]
    ]
    pattern_counts = (
        mismatches.groupby(
            ["long_delta", "short_delta", "participation_pass"], dropna=False
        )
        .size()
        .sort_index()
    )
    patterns = {
        f"long={stable_value(long)},short={stable_value(short)},participation={bool(participation)}": int(
            count
        )
        for (long, short, participation), count in pattern_counts.items()
    }
    expected_mismatches = (
        (mismatches["long_delta"] == 1)
        & (mismatches["short_delta"] == 1)
        & mismatches["participation_pass"]
    )
    return {
        "rows_checked": int(len(checked)),
        "participation_true_rows": int(checked["participation_pass"].sum()),
        "participation_false_rows": int((~checked["participation_pass"]).sum()),
        "mismatch_rows": int(len(mismatches)),
        "delta_pattern_counts": json.dumps(patterns, sort_keys=True),
        "nonparticipation_mismatch_count": int(len(nonparticipation_mismatches)),
        "is_pre_participation_score": bool(
            len(mismatches) > 0
            and expected_mismatches.all()
            and nonparticipation_mismatches.empty
        ),
    }


def classify_modern_scores(
    reconstructed: pd.DataFrame, malformed_numeric_values: int = 0
) -> str:
    """Classify a modern-schema artifact without choosing the era boundary."""
    if malformed_numeric_values:
        return "MODERN_QUARANTINED_MALFORMED_NUMERIC"
    pattern = score_pattern_summary(reconstructed)
    if not pattern["mismatch_rows"]:
        return FIVE_COMPONENT_ERA
    if pattern["is_pre_participation_score"]:
        return PRE_PARTICIPATION_ERA
    return "MODERN_QUARANTINED_SCORE_MISMATCH"


def stable_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def percentile_summary(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {key: None for key in ("min", "p05", "median", "p95", "max")}
    return {
        "min": float(clean.min()),
        "p05": float(clean.quantile(0.05)),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def values_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    if isinstance(left, (float, np.floating)) or isinstance(
        right, (float, np.floating)
    ):
        try:
            return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-12))
        except (TypeError, ValueError):
            pass
    return bool(left == right)


def changed_fields(group: pd.DataFrame, fields: list[str]) -> list[str]:
    changed: list[str] = []
    for field in fields:
        if field not in group.columns:
            continue
        values = group[field].tolist()
        if any(not values_equal(values[0], value) for value in values[1:]):
            changed.append(field)
    return changed


def fixture_in_requested_range(
    fixture: dict[str, Any],
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
) -> bool:
    date = pd.Timestamp(fixture["lower_date"])
    return not (
        (date_from is not None and date < date_from)
        or (date_to is not None and date > date_to)
    )


def run_known_case_validation(
    all_supported_rows: pd.DataFrame,
    canonical_rows: pd.DataFrame,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
) -> tuple[pd.DataFrame, list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for fixture in KNOWN_CASES:
        if not fixture_in_requested_range(fixture, date_from, date_to):
            failures.append(
                f"known case {fixture['case_id']} excluded by requested date filters"
            )
            continue
        symbol = fixture["symbol"]
        date = pd.Timestamp(fixture["lower_date"])
        duplicates = all_supported_rows[
            (all_supported_rows["symbol"] == symbol)
            & (all_supported_rows["lower_date"] == date)
        ].sort_values(["artifact_execution_utc", "source_s3_key"])
        canonical = canonical_rows[
            (canonical_rows["symbol"] == symbol)
            & (canonical_rows["lower_date"] == date)
        ]

        print(f"\n[KNOWN CASE] {fixture['case_id']} duplicates={len(duplicates)}")
        if duplicates.empty:
            failures.append(f"known case missing: {fixture['case_id']}")
            records.append(
                {
                    "case_id": fixture["case_id"],
                    "symbol": symbol,
                    "lower_date": date,
                    "status": "MISSING",
                    "is_canonical": False,
                    "assertions_pass": False,
                }
            )
            continue
        if len(canonical) != 1:
            failures.append(
                f"known case {fixture['case_id']} has {len(canonical)} canonical rows"
            )

        canonical_keys = set(canonical["source_s3_key"].astype(str))
        for _, row in duplicates.iterrows():
            is_canonical = str(row["source_s3_key"]) in canonical_keys
            print(
                f"  {row['artifact_execution_utc']} canonical={is_canonical} "
                f"long={row['reconstructed_long_score']}/{row['mtf_long_score']} "
                f"pre_long={row['pre_participation_long']} "
                f"participation={row['participation_pass']}"
            )
            assertion_messages: list[str] = []
            if is_canonical:
                for field, expected in fixture["expected"].items():
                    if bool(row[field]) != bool(expected):
                        assertion_messages.append(
                            f"{field}: expected {expected}, got {row[field]}"
                        )
                for field, (expected, tolerance) in fixture["approx"].items():
                    actual = pd.to_numeric(
                        pd.Series([row[field]]), errors="coerce"
                    ).iloc[0]
                    if pd.isna(actual) or not math.isclose(
                        float(actual),
                        float(expected),
                        rel_tol=0,
                        abs_tol=float(tolerance),
                    ):
                        assertion_messages.append(
                            f"{field}: expected approximately {expected}, got {actual}"
                        )
                if not bool(row["long_score_match"] and row["short_score_match"]):
                    assertion_messages.append(
                        "stored scores do not reconstruct exactly"
                    )
                failures.extend(
                    f"known case {fixture['case_id']}: {message}"
                    for message in assertion_messages
                )

            record = {
                "case_id": fixture["case_id"],
                "symbol": symbol,
                "lower_date": date,
                "source_s3_key": row["source_s3_key"],
                "artifact_execution_utc": row["artifact_execution_utc"],
                "is_canonical": is_canonical,
                "status": "CHECKED",
                "assertions_pass": is_canonical and not assertion_messages,
                "assertion_messages": " | ".join(assertion_messages),
            }
            for field in DERIVED_BOOL_FIELDS + [
                "lower_sig_vol_current_bar",
                "lower_spy_qqq_vol_ma_ratio",
                "middle_sig_vol_current_bar",
                "middle_spy_qqq_vol_ma_ratio",
                "reconstructed_long_score",
                "mtf_long_score",
                "reconstructed_short_score",
                "mtf_short_score",
            ]:
                record[field] = row.get(field)
            records.append(record)

    return pd.DataFrame(records), failures


def write_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    output = df.copy()
    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = pd.NA
        output = output[columns]
    output.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    date_from = parse_date(args.date_from, "--date-from")
    date_to = parse_date(args.date_to, "--date-to")
    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError("--date-from must be on or before --date-to")
    output_dir = prepare_output_dir(args.output_dir, args.overwrite_output)

    print("[SAFETY] READ-ONLY DIAGNOSTIC")
    print("[SAFETY] NO S3 WRITES")
    print("[SAFETY] NO PRODUCTION CHANGES")
    print(f"[OUTPUT] local directory: {output_dir}")

    if args.local_history_dir:
        client, sources = list_local_objects(Path(args.local_history_dir))
        source_mode = "local"
    else:
        client, sources = list_s3_objects(
            args.bucket, combo_prefix(args.prefix, args.combo)
        )
        source_mode = "s3_read_only"
    if not sources:
        raise RuntimeError("no combo-history parquet objects found")

    inventory_records: list[dict[str, Any]] = []
    artifact_frames: dict[str, pd.DataFrame] = {}
    schema_columns: dict[str, tuple[str, ...]] = {}

    for source in sources:
        print(f"[READ] {source.key}")
        raw = read_object(client, args.bucket, source)
        raw.columns = raw.columns.astype(str)
        signature = schema_signature(raw.columns)
        schema_columns.setdefault(signature, tuple(sorted(raw.columns)))
        era, missing = provisional_era(raw.columns)

        fundamental_missing = [
            field for field in ("symbol", "lower_date") if field not in raw
        ]
        malformed_lower_dates = 0
        filtered = raw.copy()
        if not fundamental_missing:
            filtered["lower_date"], malformed_lower_dates = normalize_dates(
                filtered["lower_date"]
            )
            if "middle_date" in filtered:
                filtered["middle_date"], _ = normalize_dates(filtered["middle_date"])
            if "upper_date" in filtered:
                filtered["upper_date"], _ = normalize_dates(filtered["upper_date"])
            if date_from is not None:
                filtered = filtered[filtered["lower_date"] >= date_from]
            if date_to is not None:
                filtered = filtered[filtered["lower_date"] <= date_to]
        else:
            era = "UNKNOWN_OR_MIXED"
            filtered = filtered.iloc[0:0]

        filtered = filtered.copy()
        filtered["source_s3_key"] = source.key
        filtered["artifact_execution_utc"] = source.execution_utc
        filtered["provisional_logic_era"] = era
        artifact_frames[source.key] = filtered

        lower_dates = (
            filtered["lower_date"].dropna()
            if "lower_date" in filtered
            else pd.Series(dtype="datetime64[ns]")
        )
        required_nan_counts = {
            field: int(filtered[field].isna().sum())
            for field in MODERN_REQUIRED_FIELDS
            if field in filtered.columns
        }
        inventory_records.append(
            {
                "source_s3_key": source.key,
                "artifact_execution_utc": source.execution_utc,
                "source_row_count": len(raw),
                "row_count": len(filtered),
                "column_count": len(raw.columns),
                "schema_signature": signature,
                "min_lower_date": (
                    lower_dates.min() if not lower_dates.empty else pd.NaT
                ),
                "max_lower_date": (
                    lower_dates.max() if not lower_dates.empty else pd.NaT
                ),
                "modal_lower_date": (
                    lower_dates.value_counts().sort_index().idxmax()
                    if not lower_dates.empty
                    else pd.NaT
                ),
                "unique_lower_dates": int(lower_dates.nunique()),
                "unique_symbols": (
                    int(filtered["symbol"].astype(str).nunique())
                    if "symbol" in filtered
                    else 0
                ),
                "provisional_logic_era": era,
                "missing_modern_fields": "|".join(missing),
                "missing_modern_field_count": len(missing),
                "malformed_lower_dates": malformed_lower_dates,
                "required_field_nan_count": sum(required_nan_counts.values()),
                "required_field_nan_counts": json.dumps(
                    required_nan_counts, sort_keys=True, separators=(",", ":")
                ),
                "fundamental_schema_error": bool(fundamental_missing),
                **{
                    f"has_{field}": field in raw.columns
                    for field in MODERN_REQUIRED_FIELDS
                },
            }
        )

    inventory = pd.DataFrame(inventory_records).sort_values(
        ["artifact_execution_utc", "source_s3_key"]
    )
    analyzed = inventory[inventory["row_count"] > 0]
    median_rows = float(analyzed["row_count"].median()) if not analyzed.empty else 0.0
    incomplete_floor = median_rows * 0.5
    inventory["suspiciously_incomplete"] = (
        (inventory["row_count"] > 0)
        & (median_rows > 0)
        & (inventory["row_count"] < incomplete_floor)
    )

    all_rows = pd.concat(artifact_frames.values(), ignore_index=True, sort=False)
    if "symbol" in all_rows:
        all_rows["symbol"] = all_rows["symbol"].astype(str).str.strip().str.upper()

    row_counts_by_date = (
        all_rows.groupby(["lower_date", "source_s3_key"], dropna=False)
        .size()
        .rename("artifact_rows_for_market_date")
        .reset_index()
        if not all_rows.empty and "lower_date" in all_rows
        else pd.DataFrame(
            columns=["lower_date", "source_s3_key", "artifact_rows_for_market_date"]
        )
    )
    market_execution_records: list[dict[str, Any]] = []
    if not row_counts_by_date.empty:
        execution_lookup = inventory.set_index("source_s3_key")[
            "artifact_execution_utc"
        ].to_dict()
        for lower_date, group in row_counts_by_date.groupby(
            "lower_date", sort=True, dropna=False
        ):
            ordered = group.assign(
                artifact_execution_utc=group["source_s3_key"].map(execution_lookup)
            ).sort_values(["artifact_execution_utc", "source_s3_key"])
            market_execution_records.append(
                {
                    "audit_type": "MARKET_DATE_EXECUTION_SUMMARY",
                    "lower_date": lower_date,
                    "artifact_execution_count": int(len(ordered)),
                    "execution_timestamps": "|".join(
                        str(value) for value in ordered["artifact_execution_utc"]
                    ),
                    "row_counts": "|".join(
                        str(int(value))
                        for value in ordered["artifact_rows_for_market_date"]
                    ),
                    "source_keys": "|".join(ordered["source_s3_key"].astype(str)),
                }
            )

    exceptions = pd.DataFrame(columns=EXCEPTION_FIELDS)
    era_records: list[dict[str, Any]] = []
    supported_frames: list[pd.DataFrame] = []
    validation_errors: list[str] = []
    reconstructed_by_key: dict[str, pd.DataFrame] = {}

    for _, artifact in inventory.iterrows():
        key = artifact["source_s3_key"]
        frame = artifact_frames[key]
        era = artifact["provisional_logic_era"]
        record = {
            "source_s3_key": key,
            "artifact_execution_utc": artifact["artifact_execution_utc"],
            "schema_signature": artifact["schema_signature"],
            "schema_columns": (
                "|".join(schema_columns[str(artifact["schema_signature"])])
                if args.include_unsupported_era_in_inventory
                and era != "MODERN_SUPPORTED_CANDIDATE"
                else ""
            ),
            "provisional_logic_era": era,
            "final_logic_era": era,
            "rows_checked": 0,
            "long_score_mismatches": 0,
            "short_score_mismatches": 0,
            "either_score_mismatches": 0,
            "malformed_numeric_values": 0,
            "participation_true_rows": 0,
            "participation_false_rows": 0,
            "delta_pattern_counts": "{}",
            "nonparticipation_mismatch_count": 0,
            "scoring_contract": "",
            "validation_status": "NOT_ATTEMPTED",
        }
        if (
            args.phase == "validate"
            and era == "MODERN_SUPPORTED_CANDIDATE"
            and not frame.empty
        ):
            reconstructed, malformed = reconstruct_scores(frame)
            malformed_total = sum(malformed.values())
            mismatches = reconstructed[reconstructed["either_score_mismatch"]].copy()
            pattern = score_pattern_summary(reconstructed)
            score_classification = classify_modern_scores(
                reconstructed, malformed_total
            )
            reconstructed_by_key[key] = reconstructed
            record.update(
                {
                    "rows_checked": len(reconstructed),
                    "long_score_mismatches": int(
                        (~reconstructed["long_score_match"]).sum()
                    ),
                    "short_score_mismatches": int(
                        (~reconstructed["short_score_match"]).sum()
                    ),
                    "either_score_mismatches": len(mismatches),
                    "malformed_numeric_values": malformed_total,
                    **{
                        name: pattern[name]
                        for name in (
                            "participation_true_rows",
                            "participation_false_rows",
                            "delta_pattern_counts",
                            "nonparticipation_mismatch_count",
                        )
                    },
                }
            )
            if score_classification == "MODERN_QUARANTINED_MALFORMED_NUMERIC":
                record["final_logic_era"] = "MODERN_QUARANTINED_MALFORMED_NUMERIC"
                record["validation_status"] = "QUARANTINED"
                validation_errors.append(
                    f"{key}: {malformed_total} malformed numeric values"
                )
            elif not mismatches.empty:
                if score_classification == PRE_PARTICIPATION_ERA:
                    record["final_logic_era"] = PRE_PARTICIPATION_ERA
                    record["scoring_contract"] = (
                        "four_component_without_participation_point"
                    )
                    record["validation_status"] = "HISTORICAL_EXCLUDED"
                else:
                    record["final_logic_era"] = "MODERN_QUARANTINED_SCORE_MISMATCH"
                    record["validation_status"] = "QUARANTINED_UNEXPLAINED"
                    validation_errors.append(
                        f"{key}: {len(mismatches)} unexplained score mismatches"
                    )
                exceptions = pd.concat(
                    [exceptions, mismatches.reindex(columns=EXCEPTION_FIELDS)],
                    ignore_index=True,
                )
            else:
                record["final_logic_era"] = "MODERN_EXACT_CANDIDATE"
                record["validation_status"] = "EXACT_CANDIDATE"
        elif era == "MODERN_SUPPORTED_CANDIDATE" and frame.empty:
            record["validation_status"] = "NO_ROWS_IN_DATE_RANGE"
        elif era != "MODERN_SUPPORTED_CANDIDATE":
            record["validation_status"] = "UNSUPPORTED_SCHEMA"
        era_records.append(record)

    schema_eras = pd.DataFrame(era_records).sort_values(
        ["artifact_execution_utc", "source_s3_key"]
    )

    # The supported five-component contract begins with the first exact artifact
    # after the last formally recognized pre-participation artifact. Exact modern
    # artifacts before that boundary are inventoried but never enter calibration.
    historical = schema_eras[schema_eras["final_logic_era"] == PRE_PARTICIPATION_ERA]
    historical_end = (
        historical["artifact_execution_utc"].max() if not historical.empty else None
    )
    exact_candidates = schema_eras["final_logic_era"] == "MODERN_EXACT_CANDIDATE"
    if historical_end is not None:
        supported_mask = exact_candidates & (
            schema_eras["artifact_execution_utc"] > historical_end
        )
        earlier_exact_mask = exact_candidates & ~supported_mask
        schema_eras.loc[earlier_exact_mask, "final_logic_era"] = (
            "MODERN_EXACT_BEFORE_FIVE_COMPONENT_ERA"
        )
        schema_eras.loc[earlier_exact_mask, "validation_status"] = "HISTORICAL_EXCLUDED"
    else:
        supported_mask = exact_candidates
    schema_eras.loc[supported_mask, "final_logic_era"] = FIVE_COMPONENT_ERA
    schema_eras.loc[supported_mask, "scoring_contract"] = (
        "five_component_with_participation"
    )
    schema_eras.loc[supported_mask, "validation_status"] = "EXACT"

    for key in schema_eras.loc[supported_mask, "source_s3_key"]:
        reconstructed = reconstructed_by_key[str(key)]
        reconstructed["logic_era"] = FIVE_COMPONENT_ERA
        supported_frames.append(reconstructed)

    first_supported_artifact = (
        schema_eras.loc[supported_mask, "artifact_execution_utc"].min()
        if supported_mask.any()
        else None
    )
    last_supported_artifact = (
        schema_eras.loc[supported_mask, "artifact_execution_utc"].max()
        if supported_mask.any()
        else None
    )
    era_lookup = schema_eras.set_index("source_s3_key")["final_logic_era"].to_dict()
    inventory["final_logic_era"] = inventory["source_s3_key"].map(era_lookup)

    supported_all = (
        pd.concat(supported_frames, ignore_index=True, sort=False)
        if supported_frames
        else pd.DataFrame()
    )

    incomplete_keys = set(
        inventory.loc[inventory["suspiciously_incomplete"], "source_s3_key"].astype(str)
    )
    dedupe_records: list[dict[str, Any]] = []
    canonical_indices: list[int] = []
    if not supported_all.empty:
        modal_lookup = inventory.set_index("source_s3_key")[
            "modal_lower_date"
        ].to_dict()
        supported_all["artifact_modal_lower_date"] = supported_all["source_s3_key"].map(
            modal_lookup
        )
        supported_all["artifact_suspiciously_incomplete"] = supported_all[
            "source_s3_key"
        ].isin(incomplete_keys)
        supported_all["canonical_eligible"] = (
            ~supported_all["artifact_suspiciously_incomplete"]
            & supported_all["lower_date"].notna()
            & (
                supported_all["lower_date"]
                == supported_all["artifact_modal_lower_date"]
            )
        )
        grouped = supported_all.groupby(
            ["symbol", "lower_date"], sort=True, dropna=False
        )
        for (symbol, lower_date), group in grouped:
            ordered = group.sort_values(["artifact_execution_utc", "source_s3_key"])
            eligible = ordered[ordered["canonical_eligible"]]
            canonical_index = eligible.index[-1] if not eligible.empty else None
            if canonical_index is not None:
                canonical_indices.append(canonical_index)
            changed = changed_fields(
                ordered, AUDIT_COMPARE_FIELDS + DERIVED_BOOL_FIELDS
            )
            for index, row in ordered.iterrows():
                dedupe_records.append(
                    {
                        "audit_type": "SYMBOL_DATE_EXECUTION",
                        "symbol": symbol,
                        "lower_date": lower_date,
                        "source_s3_key": row["source_s3_key"],
                        "artifact_execution_utc": row["artifact_execution_utc"],
                        "duplicate_count": len(ordered),
                        "canonical_eligible": bool(row["canonical_eligible"]),
                        "is_canonical": index == canonical_index,
                        "exclusion_reason": (
                            ""
                            if row["canonical_eligible"]
                            else (
                                "SUSPICIOUSLY_INCOMPLETE_ARTIFACT"
                                if row["artifact_suspiciously_incomplete"]
                                else "STALE_ROW_VS_ARTIFACT_MODAL_DATE"
                            )
                        ),
                        "changed_field_count": len(changed),
                        "changed_fields": "|".join(changed),
                        **{
                            field: row.get(field)
                            for field in LOWER_OHLCV + DERIVED_BOOL_FIELDS
                        },
                    }
                )

    dedupe_audit = pd.concat(
        [pd.DataFrame(market_execution_records), pd.DataFrame(dedupe_records)],
        ignore_index=True,
        sort=False,
    )
    canonical = (
        supported_all.loc[canonical_indices]
        .sort_values(
            ["lower_date", "symbol", "artifact_execution_utc", "source_s3_key"]
        )
        .reset_index(drop=True)
        if canonical_indices
        else pd.DataFrame(columns=supported_all.columns)
    )

    known_case_validation = pd.DataFrame()
    fixture_failures: list[str] = []
    if args.phase == "validate":
        known_case_validation, fixture_failures = run_known_case_validation(
            supported_all, canonical, date_from, date_to
        )
        validation_errors.extend(fixture_failures)

    market_date_counts = (
        row_counts_by_date.groupby("lower_date", dropna=True)[
            "artifact_rows_for_market_date"
        ].max()
        if not row_counts_by_date.empty
        else pd.Series(dtype="int64")
    )
    coverage = {
        "source_mode": source_mode,
        "phase": args.phase,
        "bucket": args.bucket if source_mode == "s3_read_only" else None,
        "prefix": args.prefix if source_mode == "s3_read_only" else None,
        "combo": args.combo,
        "date_from": str(date_from.date()) if date_from is not None else None,
        "date_to": str(date_to.date()) if date_to is not None else None,
        "artifact_count": len(inventory),
        "artifacts_with_rows_in_range": int((inventory["row_count"] > 0).sum()),
        "min_execution_timestamp": stable_value(
            inventory["artifact_execution_utc"].min()
        ),
        "max_execution_timestamp": stable_value(
            inventory["artifact_execution_utc"].max()
        ),
        "unique_lower_market_dates": (
            int(all_rows["lower_date"].nunique()) if "lower_date" in all_rows else 0
        ),
        "raw_row_count": int(inventory["source_row_count"].sum()),
        "rows_in_requested_range": int(len(all_rows)),
        "unique_symbols": (
            int(all_rows["symbol"].nunique()) if "symbol" in all_rows else 0
        ),
        "rows_by_schema_signature": {
            str(key): int(value)
            for key, value in all_rows.get("source_s3_key", pd.Series(dtype=str))
            .map(inventory.set_index("source_s3_key")["schema_signature"].to_dict())
            .value_counts()
            .sort_index()
            .items()
        },
        "artifacts_by_provisional_logic_era": {
            str(key): int(value)
            for key, value in inventory["provisional_logic_era"]
            .value_counts()
            .sort_index()
            .items()
        },
        "artifacts_by_final_logic_era": {
            str(key): int(value)
            for key, value in inventory["final_logic_era"]
            .value_counts()
            .sort_index()
            .items()
        },
        "per_market_date_row_count": percentile_summary(market_date_counts),
        "median_artifact_rows_in_range": median_rows,
        "suspicious_incomplete_floor": incomplete_floor,
        "suspiciously_incomplete_artifacts": int(
            inventory["suspiciously_incomplete"].sum()
        ),
        "duplicate_symbol_date_groups": (
            int(
                (
                    pd.DataFrame(dedupe_records)
                    .groupby(["symbol", "lower_date"])
                    .size()
                    > 1
                ).sum()
            )
            if dedupe_records
            else 0
        ),
        "supported_observation_count": int(len(canonical)),
        "supported_artifact_count": int(supported_mask.sum()),
        "first_supported_five_component_artifact": stable_value(
            first_supported_artifact
        ),
        "last_supported_five_component_artifact": stable_value(last_supported_artifact),
        "excluded_historical_era_observation_count": int(
            historical["rows_checked"].sum()
        ),
        "score_reconstruction_exception_count": int(len(exceptions)),
        "validation_error_count": len(validation_errors),
        "strict_schema": bool(args.strict_schema),
        "include_unsupported_era_in_inventory": bool(
            args.include_unsupported_era_in_inventory
        ),
    }

    write_csv(inventory, output_dir / "artifact_inventory.csv")
    write_csv(schema_eras, output_dir / "schema_eras.csv")
    write_csv(
        dedupe_audit,
        output_dir / "deduplication_audit.csv",
        columns=[
            "audit_type",
            "symbol",
            "lower_date",
            "artifact_execution_count",
            "execution_timestamps",
            "row_counts",
            "source_keys",
            "source_s3_key",
            "artifact_execution_utc",
            "duplicate_count",
            "canonical_eligible",
            "is_canonical",
            "exclusion_reason",
            "changed_field_count",
            "changed_fields",
        ]
        + LOWER_OHLCV
        + DERIVED_BOOL_FIELDS,
    )
    write_csv(
        exceptions, output_dir / "score_reconstruction_exceptions.csv", EXCEPTION_FIELDS
    )
    write_csv(
        known_case_validation,
        output_dir / "known_case_validation.csv",
    )
    supported_output = canonical.reindex(columns=SUPPORTED_OUTPUT_FIELDS)
    supported_output.to_parquet(
        output_dir / "supported_observations.parquet", index=False
    )
    json_dump(output_dir / "coverage_summary.json", coverage)

    report_lines = [
        "# Stock Options Participation Validation Report",
        "",
        "## Safety",
        "",
        "- READ-ONLY DIAGNOSTIC",
        "- NO S3 WRITES",
        "- NO PRODUCTION CHANGES",
        "",
        "## Coverage",
        "",
        f"- Phase: `{args.phase}`",
        f"- Artifacts inventoried: {len(inventory)}",
        f"- Rows in requested range: {len(all_rows)}",
        f"- Unique lower market dates: {coverage['unique_lower_market_dates']}",
        f"- Unique symbols: {coverage['unique_symbols']}",
        f"- Suspiciously incomplete artifacts: {coverage['suspiciously_incomplete_artifacts']}",
        "",
        "## Validation",
        "",
        f"- Canonical supported observations: {len(canonical)}",
        f"- Supported five-component artifacts: {int(supported_mask.sum())}",
        f"- First supported five-component artifact: `{stable_value(first_supported_artifact)}`",
        f"- Last supported five-component artifact: `{stable_value(last_supported_artifact)}`",
        f"- Excluded historical-era observations: {int(historical['rows_checked'].sum())}",
        f"- Score reconstruction exceptions: {len(exceptions)}",
        f"- Validation errors: {len(validation_errors)}",
        "",
        "## Logic eras",
        "",
    ]
    for era, count in sorted(coverage["artifacts_by_final_logic_era"].items()):
        report_lines.append(f"- {era}: {count} artifacts")
    report_lines.extend(["", "## Historical mismatch pattern", ""])
    if historical.empty:
        report_lines.append("- No pre-participation-score artifacts identified.")
    else:
        for _, row in historical.iterrows():
            report_lines.append(
                f"- `{row['artifact_execution_utc']}`: rows_checked={row['rows_checked']}, "
                f"participation_true_rows={row['participation_true_rows']}, "
                f"participation_false_rows={row['participation_false_rows']}, "
                f"mismatch_rows={row['either_score_mismatches']}, "
                f"delta_pattern_counts={row['delta_pattern_counts']}, "
                f"nonparticipation_mismatch_count={row['nonparticipation_mismatch_count']}"
            )
    report_lines.extend(["", "## Errors", ""])
    if validation_errors:
        report_lines.extend(f"- {error}" for error in sorted(validation_errors))
    else:
        report_lines.append("- None")
    report_lines.extend(
        [
            "",
            "## Scope",
            "",
            "This run performs inventory, schema-era classification, duplicate auditing, "
            "and (for validate phase) exact score reconstruction only. It does not perform "
            "threshold sensitivity, calibration, outcome analysis, or any production write.",
            "",
        ]
    )
    (output_dir / "validation_report.md").write_text("\n".join(report_lines))

    if args.phase == "validate" and validation_errors:
        print("\n[FAIL] validation did not pass:")
        for error in sorted(validation_errors):
            print(f"  - {error}")
        return 2
    print(f"[OK] wrote eight local artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
