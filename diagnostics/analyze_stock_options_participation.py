#!/usr/bin/env python3
"""Read-only inventory and score validation for immutable stock-options history.

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


@dataclass(frozen=True)
class ComboSpec:
    """Versioned routing contract used to reconstruct one options combo."""

    combo_id: str
    lower_timeframe: str
    middle_timeframe: str
    upper_timeframe: str
    lower_price_action_field: str
    lower_sigvol_field: str
    middle_sigvol_field: str = "middle_sig_vol_current_bar"
    lower_ratio_field: str = "lower_spy_qqq_vol_ma_ratio"
    middle_ratio_field: str = "middle_spy_qqq_vol_ma_ratio"
    upper_availability_field: str = "upper_wyckoff_stage"
    strong_with_upper: float = 0.05
    strong_without_upper: float = 0.10
    moderate: float = 0.25
    expected_admission_score: int = 5


COMBO_SPECS = {
    spec.combo_id: spec
    for spec in (
        ComboSpec(
            "stocks_c_dwm_all",
            "daily",
            "weekly",
            "monthly",
            "lower_exh_abs_pa_current_bar",
            "lower_sig_vol_current_bar",
        ),
        ComboSpec(
            "stocks_b_wmq_all",
            "weekly",
            "monthly",
            "quarterly",
            "lower_exh_abs_pa_current_bar",
            "lower_sig_vol_current_bar",
        ),
        ComboSpec(
            "stocks_a_mqy_all",
            "monthly",
            "quarterly",
            "yearly",
            "lower_exh_abs_pa_prior_bar",
            "lower_sig_vol_prior_bar",
        ),
    )
}

ETF_SCORE_FIELDS = [
    "etf_lower_primary_long_score",
    "etf_lower_primary_short_score",
    "etf_lower_secondary_long_score",
    "etf_lower_secondary_short_score",
    "etf_primary_long_score",
    "etf_primary_short_score",
    "etf_secondary_long_score",
    "etf_secondary_short_score",
]

# Fields shared by every supported modern scoring contract.  Combo-routed
# fields are deliberately added by ``required_fields`` below rather than
# keeping every possible route in this base schema.
MODERN_BASE_REQUIRED_FIELDS = [
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
    "lower_macdv_core_bull",
    "lower_macdv_core_bear",
    "lower_ttm_squeeze_pro",
    "lower_spy_qqq_vol_ma_ratio",
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

SCORING_BASE_NUMERIC_FIELDS = [
    "upper_wyckoff_stage",
    "upper_exh_abs_pa_prior_bar",
    "middle_wyckoff_stage",
    "middle_exh_abs_pa_prior_bar",
    "lower_ma_trend_bullish",
    "lower_ma_trend_bearish",
    "lower_macdv_core_bull",
    "lower_macdv_core_bear",
    "lower_ttm_squeeze_pro",
    "lower_spy_qqq_vol_ma_ratio",
    "middle_spy_qqq_vol_ma_ratio",
    "mtf_long_score",
    "mtf_short_score",
]

# Preserve all possible routing inputs in mismatch evidence.  Unlike the base
# list above, this is an output/audit inventory and does not impose a schema
# or numeric-validation requirement on any combo.
SCORING_AUDIT_FIELDS = SCORING_BASE_NUMERIC_FIELDS + [
    "lower_exh_abs_pa_current_bar",
    "lower_exh_abs_pa_prior_bar",
    "lower_sig_vol_current_bar",
    "lower_sig_vol_prior_bar",
    "middle_sig_vol_current_bar",
    "middle_sig_vol_prior_bar",
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
    + SCORING_AUDIT_FIELDS
)

FIVE_COMPONENT_ERA = "MODERN_SUPPORTED_FIVE_COMPONENT"
PRE_PARTICIPATION_ERA = "MODERN_PRE_PARTICIPATION_SCORE"
# These are regression floors from the validated 2026-09-01 snapshot, not an
# expected current-state snapshot.  Production history is immutable and grows.
VALIDATED_PHASE3_MIN_ARTIFACT_COUNT = 177
VALIDATED_PHASE3_MIN_OBSERVATION_COUNT = 399_851
VALIDATED_PHASE3_FIRST_ARTIFACT = pd.Timestamp("2025-12-30T00:10:19Z")
VALIDATED_PHASE3_MIN_LAST_ARTIFACT = pd.Timestamp("2026-09-01T01:11:13Z")

KNOWN_CASES = [
    {
        "case_id": "BBY_2026-07-08",
        "symbol": "BBY",
        "lower_date": "2026-07-08",
        "expected": {
            "reconstructed_long_score": 4,
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
        "expected": {
            "reconstructed_long_score": 1,
            "pre_participation_long": False,
            "participation_pass": False,
        },
        "approx": {},
    },
    {
        "case_id": "CAKE_2026-06-02",
        "symbol": "CAKE",
        "lower_date": "2026-06-02",
        "expected": {
            "reconstructed_long_score": 3,
            "price_action_long_pass": False,
            "pre_participation_long": False,
            "participation_pass": False,
        },
        "approx": {},
    },
    {
        "case_id": "CAKE_2026-07-24",
        "symbol": "CAKE",
        "lower_date": "2026-07-24",
        "expected": {
            "reconstructed_long_score": 4,
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
        "--phase", choices=["inventory", "validate", "phase3"], default="inventory"
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


def required_fields(spec: ComboSpec) -> list[str]:
    """Return the modern immutable-history contract for ``spec``."""
    routed = [
        spec.lower_price_action_field,
        spec.lower_sigvol_field,
        spec.middle_sigvol_field,
        spec.lower_ratio_field,
        spec.middle_ratio_field,
        spec.upper_availability_field,
    ]
    return list(dict.fromkeys(MODERN_BASE_REQUIRED_FIELDS + routed))


def numeric_fields(spec: ComboSpec) -> list[str]:
    routed = [
        spec.lower_price_action_field,
        spec.lower_sigvol_field,
        spec.middle_sigvol_field,
        spec.lower_ratio_field,
        spec.middle_ratio_field,
        spec.upper_availability_field,
    ]
    return list(dict.fromkeys(SCORING_BASE_NUMERIC_FIELDS + routed))


def provisional_era(
    columns: Iterable[Any], spec: ComboSpec | None = None
) -> tuple[str, list[str]]:
    spec = spec or COMBO_SPECS["stocks_c_dwm_all"]
    column_set = {str(column) for column in columns}
    missing = sorted(set(required_fields(spec)) - column_set)
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


def numeric_frame(
    df: pd.DataFrame, spec: ComboSpec | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    spec = spec or COMBO_SPECS["stocks_c_dwm_all"]
    numbers = pd.DataFrame(index=df.index)
    malformed: dict[str, int] = {}
    for field in numeric_fields(spec):
        converted = pd.to_numeric(df[field], errors="coerce")
        malformed[field] = int((df[field].notna() & converted.isna()).sum())
        numbers[field] = converted
    return numbers, malformed


def reconstruct_scores(
    df: pd.DataFrame, spec: ComboSpec | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    spec = spec or COMBO_SPECS["stocks_c_dwm_all"]
    numbers, malformed = numeric_frame(df, spec)
    result = df.copy()

    u = numbers[spec.upper_availability_field]
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

    price_action = numbers[spec.lower_price_action_field]
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

    strong_threshold = pd.Series(
        np.where(upper_available, spec.strong_with_upper, spec.strong_without_upper),
        index=df.index,
    )
    ls = numbers[spec.lower_sigvol_field]
    lr = numbers[spec.lower_ratio_field]
    ms = numbers[spec.middle_sigvol_field]
    mr = numbers[spec.middle_ratio_field]
    result["lower_route_pass"] = ((ls == 2) & (lr > strong_threshold)) | (
        (ls == 1) & (lr > spec.moderate)
    )
    result["middle_route_pass"] = ((ms == 2) & (mr > strong_threshold)) | (
        (ms == 1) & (mr > spec.moderate)
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


def reconstruct_final_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct options admission and the post-admission ETF guardrail."""
    result = df.copy()
    pre_long_score = (
        result[
            [
                "regime_long_pass",
                "ma_long_pass",
                "price_action_long_pass",
                "momentum_long_pass",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    pre_short_score = (
        result[
            [
                "regime_short_pass",
                "ma_short_pass",
                "price_action_short_pass",
                "momentum_short_pass",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    base = pd.Series("none", index=result.index, dtype=object)
    base.loc[pre_long_score >= 4] = "long"
    base.loc[pre_short_score >= 4] = "short"
    admitted = base.copy()
    admitted.loc[(base == "long") & (result["reconstructed_long_score"] < 5)] = "none"
    admitted.loc[(base == "short") & (result["reconstructed_short_score"] < 5)] = "none"
    result["reconstructed_pre_etf_signal"] = admitted

    def aggregate(fields: list[str]) -> pd.Series:
        values = result[fields].apply(pd.to_numeric, errors="coerce")
        return values.max(axis=1, skipna=True).where(values.notna().any(axis=1))

    long_fields = [field for field in ETF_SCORE_FIELDS if "long" in field]
    short_fields = [field for field in ETF_SCORE_FIELDS if "short" in field]
    etf_long = aggregate(long_fields)
    etf_short = aggregate(short_fields)
    final = admitted.copy()
    final.loc[(admitted == "long") & etf_short.notna() & (etf_short >= 4)] = "anti"
    final.loc[(admitted == "short") & etf_long.notna() & (etf_long >= 4)] = "anti"
    final.loc[(final == "long") & etf_long.notna() & (etf_long < 4)] = "watch"
    final.loc[(final == "short") & etf_short.notna() & (etf_short < 4)] = "watch"
    result["reconstructed_final_signal"] = final
    result["final_signal_match"] = (
        result["signal"].astype(str).str.strip().str.lower() == final
    )
    return result


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
                    actual = row[field]
                    matches = (
                        bool(actual) == expected
                        if isinstance(expected, bool)
                        else values_equal(actual, expected)
                    )
                    if not matches:
                        assertion_messages.append(
                            f"{field}: expected {expected}, got {actual}"
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


def build_etf_coverage(
    artifact_frames: dict[str, pd.DataFrame], schema_eras: pd.DataFrame
) -> pd.DataFrame:
    """Inventory stored ETF inputs without making them score prerequisites."""
    era_by_key = schema_eras.set_index("source_s3_key")["final_logic_era"].to_dict()
    records: list[dict[str, Any]] = []
    for key, frame in artifact_frames.items():
        for field in ETF_SCORE_FIELDS:
            present = field in frame
            records.append(
                {
                    "source_s3_key": key,
                    "logic_era": era_by_key.get(key, "UNKNOWN"),
                    "field": field,
                    "field_present": present,
                    "row_count": len(frame),
                    "non_null_count": int(frame[field].notna().sum()) if present else 0,
                    "null_count": (
                        int(frame[field].isna().sum()) if present else len(frame)
                    ),
                    "null_fraction": (
                        float(frame[field].isna().mean())
                        if present and len(frame)
                        else None
                    ),
                }
            )
    return pd.DataFrame(records)


def readiness_decision(
    *,
    supported_artifacts: int,
    canonical_dates: int,
    errors: list[str],
    post_etf_reconstructable: bool,
) -> tuple[str, str]:
    if supported_artifacts == 0:
        return (
            "D — NOT SUPPORTABLE FROM CURRENT IMMUTABLE HISTORY",
            "No exactly reconstructable modern five-component artifact was found.",
        )
    if errors:
        return (
            "B — REQUIRES BOUNDED CORRECTIVE VALIDATION",
            f"{len(errors)} fail-closed validation error(s) require resolution.",
        )
    if canonical_dates == 0:
        return (
            "C — PARTIAL / DESCRIPTIVE ONLY",
            "Supported artifacts exist but no canonical lower market date survived.",
        )
    if not post_etf_reconstructable:
        return (
            "C — PARTIAL / DESCRIPTIVE ONLY",
            "Participation reconstructs, but stored ETF fields cannot fully explain final signals.",
        )
    return (
        "A — READY FOR PHASE 3B ANALYSIS",
        "The modern supported era is contiguous and exact reconstruction passed.",
    )


def route_threshold(tier: pd.Series, upper_available: pd.Series) -> pd.Series:
    """Return the immutable production threshold, or NaN for an ineligible route."""
    return pd.Series(
        np.select(
            [tier.eq(1), tier.eq(2) & upper_available, tier.eq(2)],
            [0.25, 0.05, 0.10],
            default=np.nan,
        ),
        index=tier.index,
    )


def build_directional_opportunities(
    canonical: pd.DataFrame, combo: str = "stocks_c_dwm_all"
) -> pd.DataFrame:
    """Expand canonical production observations into independent LONG/SHORT records."""
    if not canonical["logic_era"].eq(FIVE_COMPONENT_ERA).all():
        raise AssertionError("unsupported era entered Phase 3")
    if combo not in COMBO_SPECS:
        raise ValueError(f"unknown ComboSpec: {combo}")
    spec = COMBO_SPECS[combo]
    if canonical.duplicated(["symbol", "lower_date"]).any():
        raise AssertionError("non-canonical duplicate entered Phase 3")
    canonical = canonical.copy()
    # Phase 3 originally analyzed D/W/M, whose routed fields are the current
    # fields.  Materialize those route aliases for slower combos rather than
    # duplicating or approximating production routing logic.
    canonical["lower_sig_vol_current_bar"] = canonical[spec.lower_sigvol_field]
    frames = []
    for direction in ("LONG", "SHORT"):
        frame = canonical.copy()
        frame["combo"] = combo
        frame["direction"] = direction
        frame["pre_participation"] = frame[
            f"pre_participation_{direction.lower()}"
        ].astype(bool)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    upper = pd.to_numeric(result["upper_wyckoff_stage"], errors="coerce").notna()
    result["upper_has_wyckoff"] = upper
    for route in ("lower", "middle"):
        tier = pd.to_numeric(result[f"{route}_sig_vol_current_bar"], errors="coerce")
        ratio = pd.to_numeric(result[f"{route}_spy_qqq_vol_ma_ratio"], errors="coerce")
        threshold = route_threshold(tier, upper)
        result[f"{route}_sigvol_tier"] = tier
        result[f"{route}_ratio"] = ratio
        result[f"{route}_threshold"] = threshold
        result[f"{route}_margin"] = ratio - threshold
        result[f"{route}_normalized_ratio"] = ratio / threshold
        result[f"{route}_participation_pass"] = threshold.notna() & ratio.gt(threshold)
        result[f"{route}_failure_reason"] = np.select(
            [
                ~tier.isin([1, 2]),
                tier.eq(1) & ~ratio.gt(0.25),
                tier.eq(2) & upper & ~ratio.gt(0.05),
                tier.eq(2) & ~upper & ~ratio.gt(0.10),
            ],
            [
                "NO_ELIGIBLE_SIGVOL_ROUTE",
                "TIER1_RATIO_NOT_ABOVE_0.25",
                "TIER2_RATIO_NOT_ABOVE_0.05",
                "TIER2_NO_UPPER_RATIO_NOT_ABOVE_0.10",
            ],
            default="PASS",
        )
    lower = result["lower_participation_pass"]
    middle = result["middle_participation_pass"]
    result["overall_participation_pass"] = lower | middle
    result["route_class"] = np.select(
        [lower & middle, lower, middle],
        ["BOTH", "LOWER_ONLY", "MIDDLE_ONLY"],
        default="NEITHER",
    )
    result["participation_only_blocker"] = (
        result["pre_participation"] & ~result["overall_participation_pass"]
    )
    result["admitted_five_component"] = (
        result["pre_participation"] & result["overall_participation_pass"]
    )
    eligible = result[["lower_threshold", "middle_threshold"]].notna()
    result["eligibility_state"] = np.where(
        eligible.any(axis=1), "ELIGIBLE_SIGVOL_ROUTE", "NO_ELIGIBLE_SIGVOL_ROUTE"
    )
    result["best_threshold_normalized_ratio"] = result[
        ["lower_normalized_ratio", "middle_normalized_ratio"]
    ].max(axis=1, skipna=True)
    result["best_absolute_margin"] = result[["lower_margin", "middle_margin"]].max(
        axis=1, skipna=True
    )
    if not result["overall_participation_pass"].eq(result["participation_pass"]).all():
        raise AssertionError(
            "Phase 3 route logic differs from reconstructed production logic"
        )
    bad = result["participation_only_blocker"] & result[
        "best_threshold_normalized_ratio"
    ].gt(1)
    if bad.any():
        raise AssertionError("blocker has best_threshold_normalized_ratio > 1")
    return result


def classify_distance(values: pd.Series, normalized: bool) -> pd.Series:
    if normalized:
        return (
            pd.cut(
                values,
                [-np.inf, 0.25, 0.5, 0.75, 0.9, 1, np.inf],
                right=False,
                labels=[
                    "< 0.25",
                    "0.25 - <0.50",
                    "0.50 - <0.75",
                    "0.75 - <0.90",
                    "0.90 - <1.00",
                    "> 1.00",
                ],
            )
            .astype(object)
            .where(values.ne(1), "1.00 exactly")
        )
    return (
        pd.cut(
            values,
            [-np.inf, -0.2, -0.1, -0.05, -0.025, -0.01, 0, np.inf],
            right=False,
            labels=[
                "<= -0.20",
                "-0.20 - <-0.10",
                "-0.10 - <-0.05",
                "-0.05 - <-0.025",
                "-0.025 - <-0.01",
                "-0.01 - <0",
                "> 0",
            ],
        )
        .astype(object)
        .where(values.ne(0), "0 exactly")
    )


def scenario_pass(
    df: pd.DataFrame, *, strong_available=0.05, strong_unavailable=0.10, moderate=0.25
) -> pd.Series:
    passes = []
    for route in ("lower", "middle"):
        tier, ratio = df[f"{route}_sigvol_tier"], df[f"{route}_ratio"]
        threshold = np.where(
            tier.eq(1),
            moderate,
            np.where(df["upper_has_wyckoff"], strong_available, strong_unavailable),
        )
        passes.append(tier.isin([1, 2]) & ratio.gt(threshold))
    return passes[0] | passes[1]


def construct_episodes(directional: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Use adjacency in global canonical-date coverage, not calendar-day gaps."""
    ordered_dates = sorted(directional["lower_date"].dropna().unique())
    positions = {date: pos for pos, date in enumerate(ordered_dates)}
    pre = (
        directional[directional["pre_participation"]]
        .copy()
        .sort_values(["symbol", "direction", "lower_date"])
    )
    pre["coverage_position"] = pre["lower_date"].map(positions)
    new = pre.groupby(["symbol", "direction"])["coverage_position"].diff().ne(1)
    pre["episode_number"] = (
        new.groupby([pre["symbol"], pre["direction"]]).cumsum().astype(int)
    )
    pre["episode_id"] = (
        pre["symbol"] + ":" + pre["direction"] + ":" + pre["episode_number"].astype(str)
    )
    rows = []
    for episode_id, group in pre.groupby("episode_id", sort=False):
        passed = group["overall_participation_pass"]
        first = group.loc[passed, "lower_date"].min() if passed.any() else pd.NaT
        rows.append(
            {
                "episode_id": episode_id,
                "symbol": group["symbol"].iloc[0],
                "direction": group["direction"].iloc[0],
                "start_date": group["lower_date"].min(),
                "end_date": group["lower_date"].max(),
                "observation_count": len(group),
                "participation_pass_count": int(passed.sum()),
                "participation_fail_count": int((~passed).sum()),
                "first_participation_pass_date": first,
                "observations_until_first_pass": (
                    int(np.flatnonzero(passed.to_numpy())[0]) + 1
                    if passed.any()
                    else pd.NA
                ),
                "ever_admitted": bool(passed.any()),
                "all_blocked": bool((~passed).all()),
                "mixed_pass_fail": bool(passed.any() and (~passed).any()),
                "block_to_pass": bool(
                    ((~passed.shift(fill_value=passed.iloc[0])) & passed).any()
                ),
                "pass_to_block": bool(
                    (passed.shift(fill_value=passed.iloc[0]) & ~passed).any()
                ),
            }
        )
    return pd.DataFrame(rows), pre.set_index(
        directional[directional["pre_participation"]]
        .sort_values(["symbol", "direction", "lower_date"])
        .index
    )["episode_id"].reindex(directional.index)


def transition_table(directional: pd.DataFrame) -> pd.DataFrame:
    pre = (
        directional[directional["pre_participation"]]
        .sort_values(["symbol", "direction", "lower_date"])
        .copy()
    )
    positions = {
        date: pos
        for pos, date in enumerate(sorted(directional["lower_date"].dropna().unique()))
    }
    pre["coverage_position"] = pre["lower_date"].map(positions)
    previous = pre.groupby(["symbol", "direction"])[
        "overall_participation_pass"
    ].shift()
    adjacent = pre.groupby(["symbol", "direction"])["coverage_position"].diff().eq(1)
    valid = previous.notna() & adjacent
    pre = pre[valid].copy()
    pre["transition"] = (
        np.where(previous[valid], "PASS", "BLOCK")
        + " -> "
        + np.where(pre["overall_participation_pass"], "PASS", "BLOCK")
    )
    pre["sigvol_branch"] = (
        "lower_tier"
        + pre["lower_sigvol_tier"].fillna(-1).astype(int).astype(str)
        + "_middle_tier"
        + pre["middle_sigvol_tier"].fillna(-1).astype(int).astype(str)
    )
    return (
        pre.groupby(["direction", "sigvol_branch", "transition"])
        .size()
        .rename("count")
        .reset_index()
        .assign(
            probability=lambda x: x["count"]
            / x.groupby(["direction", "sigvol_branch"])["count"].transform("sum")
        )
    )


def assert_phase3_population(canonical: pd.DataFrame, coverage: dict[str, Any]) -> None:
    """Fail closed on contract regressions while permitting a validated tail."""
    actual_first = pd.Timestamp(coverage["first_supported_five_component_artifact"])
    actual_last = pd.Timestamp(coverage["last_supported_five_component_artifact"])
    errors = []
    if coverage["combo"] != "stocks_c_dwm_all":
        errors.append("combo is not stocks_c_dwm_all")
    if coverage.get("scoring_contract") != "five_component_with_participation":
        errors.append("scoring contract is not five_component_with_participation")
    if coverage["supported_artifact_count"] < VALIDATED_PHASE3_MIN_ARTIFACT_COUNT:
        errors.append("supported artifact count is below validated minimum")
    if len(canonical) < VALIDATED_PHASE3_MIN_OBSERVATION_COUNT:
        errors.append("canonical observation count is below validated minimum")
    if actual_first != VALIDATED_PHASE3_FIRST_ARTIFACT:
        errors.append("first supported artifact changed")
    if actual_last < VALIDATED_PHASE3_MIN_LAST_ARTIFACT:
        errors.append("last supported artifact is before validated minimum")
    if coverage.get("validation_error_count") != 0:
        errors.append("validation errors are present")
    if not coverage.get("strict_schema"):
        errors.append("strict schema validation was not enabled")
    if not coverage.get("supported_era_contiguous"):
        errors.append("supported five-component era is not contiguous")
    if errors:
        raise AssertionError(
            "Phase 3 population invariant failed: " + ", ".join(errors)
        )


def unexplained_post_boundary_mask(
    schema_eras: pd.DataFrame,
    inventory: pd.DataFrame,
    boundary: pd.Timestamp = VALIDATED_PHASE3_FIRST_ARTIFACT,
) -> pd.Series:
    """Identify a gap/drift in the established supported production era."""
    post_boundary = schema_eras["artifact_execution_utc"] >= boundary
    supported = schema_eras["final_logic_era"].eq(FIVE_COMPONENT_ERA)
    suspicious_lookup = inventory.set_index("source_s3_key")[
        "suspiciously_incomplete"
    ].to_dict()
    understood_outlier = (
        schema_eras["source_s3_key"].map(suspicious_lookup).fillna(False).astype(bool)
    )
    return post_boundary & ~supported & ~understood_outlier


def run_phase3(
    canonical: pd.DataFrame,
    known_cases: pd.DataFrame,
    output_dir: Path,
    coverage: dict[str, Any],
) -> None:
    assert_phase3_population(canonical, coverage)
    directional = build_directional_opportunities(canonical)
    episodes, episode_ids = construct_episodes(directional)
    if not episode_ids.index.equals(directional.index):
        raise AssertionError(
            "Phase 3 episode identifiers are not aligned to the directional population"
        )
    directional["episode_id"] = episode_ids
    pre = directional[directional["pre_participation"]].copy()
    if pre["episode_id"].isna().any():
        raise AssertionError(
            "Phase 3 episode identifiers are missing from pre-participation rows"
        )
    episode_owners = pre.groupby("episode_id")[["symbol", "direction"]].nunique()
    if episode_owners.gt(1).any(axis=None):
        raise AssertionError(
            "Phase 3 episode identifiers are not globally unique by symbol and direction"
        )

    population = []
    for direction, group in directional.groupby("direction"):
        pre_count = int(group["pre_participation"].sum())
        for population_name, mask in {
            "ALL_CANONICAL_DIRECTIONAL": pd.Series(True, index=group.index),
            "PRE_PARTICIPATION": group["pre_participation"],
            "ADMITTED": group["admitted_five_component"],
            "PARTICIPATION_ONLY_BLOCKER": group["participation_only_blocker"],
        }.items():
            count = int(mask.sum())
            population.append(
                {
                    "direction": direction,
                    "population": population_name,
                    "count": count,
                    "pct_all_directional": count / len(group) if len(group) else np.nan,
                    "pct_pre_participation": count / pre_count if pre_count else np.nan,
                }
            )
    population = pd.DataFrame(population)

    monthly = (
        pre.assign(calendar_month=pre["lower_date"].dt.to_period("M").astype(str))
        .groupby(["calendar_month", "direction"])
        .agg(
            pre_participation_count=("pre_participation", "size"),
            admitted_count=("overall_participation_pass", "sum"),
            median_lower_ratio=("lower_ratio", "median"),
            median_middle_ratio=("middle_ratio", "median"),
        )
        .reset_index()
    )
    monthly["blocker_count"] = (
        monthly["pre_participation_count"] - monthly["admitted_count"]
    )
    monthly["participation_pass_rate"] = (
        monthly["admitted_count"] / monthly["pre_participation_count"]
    )
    route_outcomes = (
        pre.groupby(
            [
                "direction",
                "route_class",
                "upper_has_wyckoff",
                "lower_sigvol_tier",
                "middle_sigvol_tier",
                "lower_failure_reason",
                "middle_failure_reason",
            ],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )

    route_rows = []
    bucket_rows = []
    ratio_edges = [-np.inf, 0.01, 0.025, 0.05, 0.10, 0.15, 0.25, 0.50, np.inf]
    ratio_labels = [
        "< 0.01",
        "0.01 - <0.025",
        "0.025 - <0.05",
        "0.05 - <0.10",
        "0.10 - <0.15",
        "0.15 - <0.25",
        "0.25 - <0.50",
        ">= 0.50",
    ]
    for route in ("LOWER", "MIDDLE"):
        r = route.lower()
        eligible = pre[pre[f"{r}_sigvol_tier"].isin([1, 2])].copy()
        for keys, group in eligible.groupby(
            ["direction", f"{r}_sigvol_tier", "upper_has_wyckoff"]
        ):
            values, threshold = group[f"{r}_ratio"], group[f"{r}_threshold"]
            quantiles = values.quantile(
                [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            )
            route_rows.append(
                {
                    "direction": keys[0],
                    "route": route,
                    "sigvol_tier": keys[1],
                    "upper_wyckoff": "AVAILABLE" if keys[2] else "UNAVAILABLE",
                    "count": len(group),
                    "min": values.min(),
                    **{
                        f"p{int(q*100):02d}" if q != 0.5 else "median": quantiles[q]
                        for q in quantiles.index
                    },
                    "max": values.max(),
                    "count_below_threshold": int(values.lt(threshold).sum()),
                    "count_equal_threshold": int(values.eq(threshold).sum()),
                    "count_above_threshold": int(values.gt(threshold).sum()),
                }
            )
            buckets = pd.cut(
                values, ratio_edges, right=False, labels=ratio_labels
            ).value_counts(sort=False)
            bucket_rows.extend(
                {
                    "direction": keys[0],
                    "route": route,
                    "sigvol_tier": keys[1],
                    "upper_wyckoff": "AVAILABLE" if keys[2] else "UNAVAILABLE",
                    "interval": str(label),
                    "count": int(count),
                }
                for label, count in buckets.items()
            )
    ratios, ratio_buckets = pd.DataFrame(route_rows), pd.DataFrame(bucket_rows)

    blocked = pre[~pre["overall_participation_pass"]].copy()
    distance = pd.concat(
        [
            classify_distance(blocked["best_threshold_normalized_ratio"], True)
            .value_counts(dropna=False)
            .rename_axis("bucket")
            .reset_index(name="count")
            .assign(metric="best_threshold_normalized_ratio"),
            classify_distance(blocked["best_absolute_margin"], False)
            .value_counts(dropna=False)
            .rename_axis("bucket")
            .reset_index(name="count")
            .assign(metric="best_absolute_margin"),
        ],
        ignore_index=True,
    )

    def sensitivity_record(
        direction: str,
        group: pd.DataFrame,
        family: str,
        value: float,
        passed: pd.Series,
        **extra: Any,
    ) -> dict[str, Any]:
        current = group["overall_participation_pass"]
        new = passed & ~current
        return {
            "direction": direction,
            "threshold_family": family,
            "threshold": value,
            **extra,
            "pre_participation_opportunities": len(group),
            "current_admitted": int(current.sum()),
            "scenario_admitted": int(passed.sum()),
            "incremental_admitted": int(passed.sum() - current.sum()),
            "incremental_admitted_pct_of_prepart": (passed.sum() - current.sum())
            / len(group),
            "relative_change_vs_current_admitted": (
                (passed.sum() - current.sum()) / current.sum()
                if current.sum()
                else np.nan
            ),
            "newly_admitted_symbols": int(group.loc[new, "symbol"].nunique()),
            "newly_admitted_episodes": int(group.loc[new, "episode_id"].nunique()),
        }

    one_d = []
    for direction, group in pre.groupby("direction"):
        for family, values in [
            (
                "STRONG_UPPER_AVAILABLE",
                [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10],
            ),
            (
                "STRONG_UPPER_UNAVAILABLE",
                [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10],
            ),
            ("MODERATE", [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]),
        ]:
            for value in values:
                kwargs = (
                    {"strong_available": value}
                    if family.endswith("AVAILABLE") and "UNAVAILABLE" not in family
                    else (
                        {"strong_unavailable": value}
                        if family.endswith("UNAVAILABLE")
                        else {"moderate": value}
                    )
                )
                one_d.append(
                    sensitivity_record(
                        direction, group, family, value, scenario_pass(group, **kwargs)
                    )
                )
    one_d = pd.DataFrame(one_d)
    two_d = []
    for direction, group in pre.groupby("direction"):
        for strong in [0.025, 0.05, 0.075, 0.10]:
            for moderate in [0.05, 0.10, 0.15, 0.20, 0.25]:
                passed = scenario_pass(
                    group,
                    strong_available=strong,
                    strong_unavailable=strong * 2,
                    moderate=moderate,
                )
                two_d.append(
                    sensitivity_record(
                        direction,
                        group,
                        "STRONG_VS_MODERATE",
                        strong,
                        passed,
                        strong_threshold=strong,
                        strong_unavailable_threshold=strong * 2,
                        moderate_threshold=moderate,
                    )
                )
    two_d = pd.DataFrame(two_d)

    tier_rows = []
    for tier, label in [(1, "MODERATE_TIER_1"), (2, "STRONG_TIER_2")]:
        for direction, group in pre.groupby("direction"):
            parts = []
            for r in ("lower", "middle"):
                part = group[group[f"{r}_sigvol_tier"].eq(tier)].copy()
                part["ratio"], part["threshold"], part["route_pass"] = (
                    part[f"{r}_ratio"],
                    part[f"{r}_threshold"],
                    part[f"{r}_participation_pass"],
                )
                parts.append(part)
            routes = pd.concat(parts)
            normalized = routes["ratio"] / routes["threshold"]
            tier_rows.append(
                {
                    "direction": direction,
                    "sigvol_tier": label,
                    "eligible_route_observations": len(routes),
                    "route_pass_rate": routes["route_pass"].mean(),
                    "opportunity_level_overall_pass_rate": group.loc[
                        group["lower_sigvol_tier"].eq(tier)
                        | group["middle_sigvol_tier"].eq(tier),
                        "overall_participation_pass",
                    ].mean(),
                    "ratio_median": routes["ratio"].median(),
                    "ratio_p75": routes["ratio"].quantile(0.75),
                    "ratio_p90": routes["ratio"].quantile(0.9),
                    "ratio_p95": routes["ratio"].quantile(0.95),
                    "median_normalized_ratio": normalized.median(),
                    "share_within_10pct_of_threshold": normalized.between(
                        0.9, 1, inclusive="both"
                    ).mean(),
                    "share_within_25pct_of_threshold": normalized.between(
                        0.75, 1, inclusive="both"
                    ).mean(),
                }
            )
    tiers = pd.DataFrame(tier_rows)

    symbols = (
        pre.groupby(["symbol", "direction"])
        .agg(
            pre_participation_opportunity_count=("symbol", "size"),
            blocker_count=("participation_only_blocker", "sum"),
            admitted_count=("admitted_five_component", "sum"),
            median_lower_ratio=("lower_ratio", "median"),
            median_middle_ratio=("middle_ratio", "median"),
            dominant_route_state=("route_class", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )
    symbols["blocker_rate"] = (
        symbols["blocker_count"] / symbols["pre_participation_opportunity_count"]
    )
    for name, route in [
        ("lower", "lower_sigvol_tier"),
        ("middle", "middle_sigvol_tier"),
    ]:
        dominant = pre.groupby(["symbol", "direction"])[route].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        symbols[f"dominant_{name}_sigvol_tier"] = symbols.set_index(
            ["symbol", "direction"]
        ).index.map(dominant)
    dates = blocked.groupby(["symbol", "direction"])["lower_date"].agg(
        first_blocker_date="min", last_blocker_date="max"
    )
    symbols = symbols.join(dates, on=["symbol", "direction"])

    transitions = transition_table(directional)
    outputs = {
        "population_summary.csv": population,
        "monthly_participation_summary.csv": monthly,
        "route_outcomes.csv": route_outcomes,
        "ratio_distributions.csv": ratios,
        "ratio_buckets.csv": ratio_buckets,
        "distance_to_threshold.csv": distance,
        "threshold_sensitivity_1d.csv": one_d,
        "threshold_sensitivity_2d.csv": two_d,
        "sigvol_tier_comparison.csv": tiers,
        "symbol_concentration.csv": symbols,
        "episode_summary.csv": episodes,
        "transition_summary.csv": transitions,
        "known_case_validation.csv": known_cases,
    }
    for name, frame in outputs.items():
        write_csv(frame, output_dir / name)
    concentration = {
        kind: {
            f"top_{n}_share": (
                float(symbols.nlargest(n, kind)[kind].sum() / symbols[kind].sum())
                if symbols[kind].sum()
                else None
            )
            for n in (10, 25, 50, 100)
        }
        for kind in ("blocker_count", "admitted_count")
    }
    summary = {
        "validated_population": {
            "combo": coverage["combo"],
            "scoring_contract": "five_component_with_participation",
            "supported_artifacts": coverage["supported_artifact_count"],
            "canonical_observations": len(canonical),
            "first_supported_artifact": coverage[
                "first_supported_five_component_artifact"
            ],
            "last_supported_artifact": coverage[
                "last_supported_five_component_artifact"
            ],
            "run_timestamp_utc": coverage["run_timestamp_utc"],
            "unique_market_dates": int(canonical["lower_date"].nunique()),
            "unique_symbols": int(canonical["symbol"].nunique()),
            "population_statement": (
                "Analysis uses the complete validated five-component production "
                "history available at this run."
            ),
        },
        "safety_assertions": "PASSED",
        "symbol_concentration": concentration,
        "decision_gate": "A. ENOUGH EVIDENCE FOR THRESHOLD POLICY REVIEW",
    }
    json_dump(output_dir / "phase3_summary.json", summary)
    sections = [
        "Executive finding",
        "Validated population used",
        "Current participation rule",
        "Pre-participation opportunity counts",
        "Current admission/block rates",
        "Lower vs middle route behavior",
        "Sigvol tier 1 vs tier 2 comparison",
        "Ratio distributions",
        "Distance-to-threshold findings",
        "Threshold sensitivity",
        "Symbol concentration",
        "Persistence / episode behavior",
        "Time stability",
        "Known-case validation",
        "Data limitations",
        "What this analysis CAN support",
        "What this analysis CANNOT support",
        "Decision gate",
    ]
    report = [
        "# Phase 3 Participation-Gate Diagnostic",
        "",
        "This report is descriptive evidence only. It makes no production threshold recommendation.",
        "",
    ]
    for section in sections:
        if section == "Validated population used":
            detail = (
                "Analysis uses the complete validated five-component production "
                "history available at this run. Population metadata is recorded in "
                "`phase3_summary.json`."
            )
        else:
            detail = (
                "A. ENOUGH EVIDENCE FOR THRESHOLD POLICY REVIEW"
                if section == "Decision gate"
                else "See the corresponding CSV and `phase3_summary.json` for auditable results."
            )
        report += [
            f"## {section}",
            "",
            detail,
            "",
        ]
    (output_dir / "phase3_report.md").write_text("\n".join(report))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.combo not in COMBO_SPECS:
        raise ValueError(
            f"unsupported options combo {args.combo!r}; choose from "
            f"{', '.join(sorted(COMBO_SPECS))}"
        )
    spec = COMBO_SPECS[args.combo]
    if args.phase == "phase3" and args.combo != "stocks_c_dwm_all":
        raise ValueError("Phase 3 analysis remains restricted to stocks_c_dwm_all")
    combo_required_fields = required_fields(spec)
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
        era, missing = provisional_era(raw.columns, spec)

        fundamental_missing = [
            field for field in ("symbol", "lower_date") if field not in raw
        ]
        malformed_lower_dates = 0
        malformed_date_counts = {
            field: 0 for field in ("lower_date", "middle_date", "upper_date")
        }
        filtered = raw.copy()
        if not fundamental_missing:
            filtered["lower_date"], malformed_lower_dates = normalize_dates(
                filtered["lower_date"]
            )
            malformed_date_counts["lower_date"] = malformed_lower_dates
            if "middle_date" in filtered:
                filtered["middle_date"], malformed_date_counts["middle_date"] = (
                    normalize_dates(filtered["middle_date"])
                )
            if "upper_date" in filtered:
                filtered["upper_date"], malformed_date_counts["upper_date"] = (
                    normalize_dates(filtered["upper_date"])
                )
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
            for field in combo_required_fields
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
                "malformed_required_date_values": sum(malformed_date_counts.values()),
                "malformed_required_date_counts": json.dumps(
                    malformed_date_counts, sort_keys=True, separators=(",", ":")
                ),
                "required_field_nan_count": sum(required_nan_counts.values()),
                "required_field_nan_counts": json.dumps(
                    required_nan_counts, sort_keys=True, separators=(",", ":")
                ),
                "fundamental_schema_error": bool(fundamental_missing),
                **{
                    f"has_{field}": field in raw.columns
                    for field in combo_required_fields
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
            "malformed_date_values": int(artifact["malformed_required_date_values"]),
            "participation_true_rows": 0,
            "participation_false_rows": 0,
            "delta_pattern_counts": "{}",
            "nonparticipation_mismatch_count": 0,
            "final_signal_rows_checked": 0,
            "final_signal_mismatches": 0,
            "scoring_contract": "",
            "validation_status": "NOT_ATTEMPTED",
        }
        if (
            args.phase in ("validate", "phase3")
            and era == "MODERN_SUPPORTED_CANDIDATE"
            and not frame.empty
            and not record["malformed_date_values"]
        ):
            reconstructed, malformed = reconstruct_scores(frame, spec)
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
                if all(field in reconstructed for field in ETF_SCORE_FIELDS):
                    reconstructed = reconstruct_final_signals(reconstructed)
                    reconstructed_by_key[key] = reconstructed
                    final_mismatches = int((~reconstructed["final_signal_match"]).sum())
                    record["final_signal_rows_checked"] = len(reconstructed)
                    record["final_signal_mismatches"] = final_mismatches
                    if final_mismatches:
                        record["final_logic_era"] = (
                            "MODERN_QUARANTINED_FINAL_SIGNAL_MISMATCH"
                        )
                        record["validation_status"] = "QUARANTINED_UNEXPLAINED"
                        validation_errors.append(
                            f"{key}: {final_mismatches} unexplained final signal mismatches"
                        )
                        era_records.append(record)
                        continue
                record["final_logic_era"] = "MODERN_EXACT_CANDIDATE"
                record["validation_status"] = "EXACT_CANDIDATE"
        elif era == "MODERN_SUPPORTED_CANDIDATE" and record["malformed_date_values"]:
            record["final_logic_era"] = "MODERN_QUARANTINED_MALFORMED_DATE"
            record["validation_status"] = "QUARANTINED"
            validation_errors.append(
                f"{key}: {record['malformed_date_values']} malformed required date values"
            )
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

    # Once the pinned five-component boundary has been reached, every normal
    # artifact must remain on that exact schema/scoring contract.  The existing
    # suspicious-incomplete rule is the sole operational-outlier exception.
    combo_boundary = (
        VALIDATED_PHASE3_FIRST_ARTIFACT
        if args.combo == "stocks_c_dwm_all"
        else (
            schema_eras.loc[supported_mask, "artifact_execution_utc"].min()
            if supported_mask.any()
            else None
        )
    )
    unexplained_post_boundary = (
        unexplained_post_boundary_mask(schema_eras, inventory, combo_boundary)
        if combo_boundary is not None
        else pd.Series(False, index=schema_eras.index)
    )
    for _, artifact in schema_eras.loc[unexplained_post_boundary].iterrows():
        validation_errors.append(
            f"{artifact['source_s3_key']}: post-boundary artifact is "
            f"{artifact['final_logic_era']} ({artifact['validation_status']})"
        )

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
    if args.phase in ("validate", "phase3"):
        if args.combo == "stocks_c_dwm_all":
            known_case_validation, fixture_failures = run_known_case_validation(
                supported_all, canonical, date_from, date_to
            )
            validation_errors.extend(fixture_failures)

    if args.phase == "validate" and not supported_mask.any():
        validation_errors.append(
            "no exactly reconstructable modern five-component artifact was found"
        )

    market_date_counts = (
        row_counts_by_date.groupby("lower_date", dropna=True)[
            "artifact_rows_for_market_date"
        ].max()
        if not row_counts_by_date.empty
        else pd.Series(dtype="int64")
    )
    coverage = {
        "run_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_mode": source_mode,
        "phase": args.phase,
        "bucket": args.bucket if source_mode == "s3_read_only" else None,
        "prefix": args.prefix if source_mode == "s3_read_only" else None,
        "combo": args.combo,
        "combo_spec": {
            key: stable_value(value) for key, value in spec.__dict__.items()
        },
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
        "canonical_lower_market_dates": (
            int(canonical["lower_date"].nunique()) if "lower_date" in canonical else 0
        ),
        "canonical_observations_before_deduplication": int(len(supported_all)),
        "canonical_observations_after_deduplication": int(len(canonical)),
        "symbol_union_count": (
            int(all_rows["symbol"].nunique()) if "symbol" in all_rows else 0
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
        "scoring_contract": "five_component_with_participation",
        "supported_era_contiguous": not bool(unexplained_post_boundary.any()),
        "unexplained_post_boundary_artifact_count": int(
            unexplained_post_boundary.sum()
        ),
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

    modern_candidates = schema_eras[
        schema_eras["provisional_logic_era"] == "MODERN_SUPPORTED_CANDIDATE"
    ]
    coverage["first_modern_candidate_artifact"] = stable_value(
        modern_candidates["artifact_execution_utc"].min()
        if not modern_candidates.empty
        else None
    )
    coverage["raw_artifact_executions"] = len(inventory)
    coverage["duplicate_same_market_date_groups"] = sum(
        int(record.get("artifact_execution_count", 0) > 1)
        for record in market_execution_records
    )

    etf_coverage = build_etf_coverage(artifact_frames, schema_eras)
    supported_keys = set(schema_eras.loc[supported_mask, "source_s3_key"].astype(str))
    supported_etf = etf_coverage[etf_coverage["source_s3_key"].isin(supported_keys)]
    all_etf_fields_present = bool(
        supported_keys
        and not supported_etf.empty
        and supported_etf.groupby("source_s3_key")["field_present"].all().all()
    )
    coverage["etf_fields_complete_in_supported_era"] = all_etf_fields_present
    coverage["post_etf_signal_reconstructable"] = all_etf_fields_present

    decision, decision_reason = readiness_decision(
        supported_artifacts=int(supported_mask.sum()),
        canonical_dates=int(coverage["canonical_lower_market_dates"]),
        errors=validation_errors,
        post_etf_reconstructable=all_etf_fields_present,
    )
    coverage["readiness_decision"] = decision
    coverage["readiness_reason"] = decision_reason

    write_csv(inventory, output_dir / "artifact_inventory.csv")
    write_csv(schema_eras, output_dir / "schema_eras.csv")
    schema_era_summary = (
        schema_eras.groupby(
            ["schema_signature", "final_logic_era", "validation_status"],
            dropna=False,
        )
        .agg(
            artifact_count=("source_s3_key", "size"),
            first_artifact=("artifact_execution_utc", "min"),
            last_artifact=("artifact_execution_utc", "max"),
            rows_checked=("rows_checked", "sum"),
            score_mismatches=("either_score_mismatches", "sum"),
            final_signal_mismatches=("final_signal_mismatches", "sum"),
        )
        .reset_index()
    )
    write_csv(schema_era_summary, output_dir / "schema_era_summary.csv")
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
    score_summary_columns = [
        "source_s3_key",
        "artifact_execution_utc",
        "final_logic_era",
        "validation_status",
        "rows_checked",
        "long_score_mismatches",
        "short_score_mismatches",
        "either_score_mismatches",
        "malformed_numeric_values",
        "participation_true_rows",
        "participation_false_rows",
        "delta_pattern_counts",
        "final_signal_rows_checked",
        "final_signal_mismatches",
    ]
    write_csv(
        schema_eras,
        output_dir / "score_reconstruction_summary.csv",
        score_summary_columns,
    )
    write_csv(etf_coverage, output_dir / "etf_coverage_summary.csv")
    write_csv(
        pd.DataFrame(
            [
                {"combo": args.combo, "error_number": index, "error": error}
                for index, error in enumerate(validation_errors, start=1)
            ],
            columns=["combo", "error_number", "error"],
        ),
        output_dir / "validation_errors.csv",
    )
    canonicalization = {
        "combo": args.combo,
        "raw_artifact_executions": len(inventory),
        "duplicate_same_market_date_groups": coverage[
            "duplicate_same_market_date_groups"
        ],
        "canonical_market_dates": coverage["canonical_lower_market_dates"],
        "observations_before_canonicalization": len(supported_all),
        "observations_after_canonicalization": len(canonical),
        "suspiciously_incomplete_artifacts": coverage[
            "suspiciously_incomplete_artifacts"
        ],
        "rule": "latest valid execution per symbol/lower_date",
    }
    json_dump(output_dir / "canonicalization_summary.json", canonicalization)
    write_csv(
        pd.DataFrame([canonicalization]),
        output_dir / "canonicalization_summary.csv",
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

    if args.phase == "phase3" and not validation_errors:
        run_phase3(canonical, known_case_validation, output_dir, coverage)

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
        f"- ETF fields complete in supported era: {all_etf_fields_present}",
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
            "## Readiness decision",
            "",
            f"**{decision}**",
            "",
            decision_reason,
            "",
        ]
    )
    (output_dir / "validation_report.md").write_text("\n".join(report_lines))

    if args.phase in ("validate", "phase3") and validation_errors:
        print("\n[FAIL] validation did not pass:")
        for error in sorted(validation_errors):
            print(f"  - {error}")
        return 2
    print(f"[OK] wrote local validation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
