#!/usr/bin/env python3
"""Read-only, historical-as-of scoring lineage collector.

Only S3 list/get APIs are used.  Rolling objects are read at the newest object
version whose ``LastModified`` is no later than the selected combo execution.
When that historical version is unavailable, the evidence is marked unavailable;
the current object is never silently substituted.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import yaml
from botocore.exceptions import ClientError

from indicators.core import INDICATOR_FUNCS

COMBO_ID = "stocks_c_dwm_all"
ROLE_TIMEFRAMES = {"lower": "daily", "middle": "weekly", "upper": "monthly"}
INDICATORS = (
    "ma_trend_bullish",
    "ma_trend_bearish",
    "exh_abs_pa_current_bar",
    "exh_abs_pa_prior_bar",
    "wyckoff_stage",
    "macdv_core_bull",
    "macdv_core_bear",
    "macdv_guard",
    "ttm_squeeze_pro",
    "sig_vol_current_bar",
)
HUMAN_BAR_ROWS = 130
HISTORY_TS = re.compile(r"_asof=(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.parquet$")


@dataclass(frozen=True)
class S3Evidence:
    logical_name: str
    s3_key: str
    version_id: str | None
    last_modified_utc: str | None
    evidence_status: str
    selection_reason: str
    artifact_timestamp_utc: str | None = None
    observation_timestamp: str | None = None
    maximum_bar_timestamp: str | None = None


def parse_market_date(value: str) -> date:
    return date.fromisoformat(value)


def _timestamp_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.to_datetime(index, errors="coerce"), index=index)


def filter_as_of(frame: pd.DataFrame, market_date: date) -> pd.DataFrame:
    """Return only observations whose tz-naive/local date is <= market_date."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[] if frame is None else frame.columns)
    timestamps = _timestamp_series(frame.index)
    mask = timestamps.dt.date <= market_date
    return frame.loc[mask.fillna(False)].sort_index().copy()


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
            return False
        return len(left) == len(right) and all(
            values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return bool(np.isclose(float(left), float(right), equal_nan=True))
    return left == right


def comparison(left: Any, right: Any, *, available: bool = True) -> str:
    if not available:
        return "unavailable"
    return "match" if values_equal(left, right) else "mismatch"


def timestamp_comparison(left: Any, right: Any, *, available: bool) -> str:
    if not available:
        return "unavailable"
    left_ts = pd.to_datetime(left, errors="coerce")
    right_ts = pd.to_datetime(right, errors="coerce")
    if pd.isna(left_ts) or pd.isna(right_ts):
        return "mismatch"
    return "match" if left_ts == right_ts else "mismatch"


def compare_close_lineage(
    timeframe: str,
    canonical_value: Any,
    canonical_timestamp: Any,
    snapshot_value: Any,
    snapshot_timestamp: Any,
    combo_value: Any,
    combo_timestamp: Any,
    *,
    canonical_available: bool,
    snapshot_available: bool,
    combo_available: bool,
) -> dict[str, Any]:
    canonical_snapshot_available = canonical_available and snapshot_available
    snapshot_combo_available = snapshot_available and combo_available
    canonical_snapshot_time = timestamp_comparison(
        canonical_timestamp,
        snapshot_timestamp,
        available=canonical_snapshot_available,
    )
    snapshot_combo_time = timestamp_comparison(
        snapshot_timestamp, combo_timestamp, available=snapshot_combo_available
    )
    first_issue = "none"
    if not canonical_available:
        first_issue = "canonical_unavailable"
    elif not snapshot_available:
        first_issue = "snapshot_unavailable"
    elif comparison(canonical_value, snapshot_value) == "mismatch":
        first_issue = "canonical_to_snapshot"
    elif canonical_snapshot_time == "mismatch":
        first_issue = "canonical_to_snapshot_timestamp"
    elif not combo_available:
        first_issue = "combo_unavailable"
    elif comparison(snapshot_value, combo_value) == "mismatch":
        first_issue = "snapshot_to_combo"
    elif snapshot_combo_time == "mismatch":
        first_issue = "snapshot_to_combo_timestamp"
    return {
        "timeframe": timeframe,
        "canonical_close": canonical_value,
        "canonical_timestamp": canonical_timestamp,
        "snapshot_close": snapshot_value,
        "snapshot_timestamp": snapshot_timestamp,
        "combo_close": combo_value,
        "combo_timestamp": combo_timestamp,
        "canonical_to_snapshot": comparison(
            canonical_value, snapshot_value, available=canonical_snapshot_available
        ),
        "snapshot_to_combo": comparison(
            snapshot_value, combo_value, available=snapshot_combo_available
        ),
        "canonical_to_snapshot_timestamp": canonical_snapshot_time,
        "snapshot_to_combo_timestamp": snapshot_combo_time,
        "first_issue": first_issue,
    }


def compare_indicator_lineage(
    timeframe: str,
    indicator: str,
    recomputed: Any,
    snapshot: Any,
    combo: Any,
    *,
    recomputed_available: bool,
    snapshot_available: bool,
    combo_available: bool,
) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "indicator": indicator,
        "recomputed_value": recomputed,
        "snapshot_value": snapshot,
        "combo_value": combo,
        "recomputed_to_snapshot": comparison(
            recomputed,
            snapshot,
            available=recomputed_available and snapshot_available,
        ),
        "snapshot_to_combo": comparison(
            snapshot, combo, available=snapshot_available and combo_available
        ),
    }


class ReadOnlyS3:
    """Narrow S3 reader: this class deliberately exposes no mutation method."""

    def __init__(self, bucket: str, prefix: str, client: Any | None = None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3")
        self.version_errors: dict[str, str] = {}

    def key(self, relative: str) -> str:
        relative = relative.lstrip("/")
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def list_keys(self, relative_prefix: str) -> list[dict[str, Any]]:
        prefix = self.key(relative_prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        return [
            item
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            for item in page.get("Contents", [])
        ]

    def versions(self, key: str) -> list[dict[str, Any]]:
        try:
            paginator = self.client.get_paginator("list_object_versions")
            return [
                item
                for page in paginator.paginate(Bucket=self.bucket, Prefix=key)
                for item in page.get("Versions", [])
                if item.get("Key") == key
            ]
        except ClientError as exc:
            self.version_errors[key] = (
                f"{exc.response.get('Error', {}).get('Code', 'ClientError')}: {exc}"
            )
            return []

    def get_parquet(self, key: str, version_id: str | None = None) -> pd.DataFrame:
        args = {"Bucket": self.bucket, "Key": key}
        if version_id:
            args["VersionId"] = version_id
        body = self.client.get_object(**args)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))

    def get_version_as_of(
        self, relative: str, cutoff: datetime, logical_name: str
    ) -> tuple[pd.DataFrame | None, S3Evidence]:
        key = self.key(relative)
        eligible = [
            version
            for version in self.versions(key)
            if version["LastModified"].astimezone(timezone.utc) <= cutoff
        ]
        if not eligible:
            version_error = self.version_errors.get(key)
            reason = (
                f"object-version listing unavailable ({version_error}); latest not substituted"
                if version_error
                else "no object version existed at or before combo execution; latest not substituted"
            )
            return None, S3Evidence(
                logical_name,
                key,
                None,
                None,
                "unavailable",
                reason,
            )
        chosen = max(eligible, key=lambda item: item["LastModified"])
        modified = chosen["LastModified"].astimezone(timezone.utc)
        return self.get_parquet(key, chosen.get("VersionId")), S3Evidence(
            logical_name,
            key,
            chosen.get("VersionId"),
            modified.isoformat(),
            "available",
            "latest object version no later than combo execution",
        )


def _history_execution(key: str) -> datetime | None:
    match = HISTORY_TS.search(key)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S").replace(
        tzinfo=timezone.utc
    )


def find_historical_combo(
    reader: ReadOnlyS3, symbol: str, market_date: date
) -> tuple[pd.DataFrame | None, S3Evidence, datetime | None]:
    relative = f"combo_history/stocks/{COMBO_ID}/"
    candidates: list[tuple[datetime, str, dict[str, Any]]] = []
    for item in reader.list_keys(relative):
        execution = _history_execution(item["Key"])
        if execution and execution.date() in {
            market_date,
            market_date + timedelta(days=1),
            market_date + timedelta(days=2),
        }:
            candidates.append((execution, item["Key"], item))

    matches: list[tuple[datetime, str, dict[str, Any], pd.DataFrame]] = []
    for execution, key, item in sorted(candidates):
        frame = reader.get_parquet(key)
        if "symbol" not in frame.columns or "lower_date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["lower_date"], errors="coerce").dt.date
        rows = frame.loc[
            frame["symbol"].astype(str).str.upper().eq(symbol) & dates.eq(market_date)
        ].copy()
        if not rows.empty:
            matches.append((execution, key, item, rows))

    if not matches:
        key = reader.key(relative)
        return (
            None,
            S3Evidence(
                "historical_combo",
                key,
                None,
                None,
                "unavailable",
                "no immutable combo-history row matched symbol and lower market date",
            ),
            None,
        )

    # The first actual production observation for this market date is authoritative.
    execution, key, item, rows = min(matches, key=lambda value: value[0])
    evidence = S3Evidence(
        "historical_combo",
        key,
        None,
        item["LastModified"].astimezone(timezone.utc).isoformat(),
        "available",
        "earliest immutable combo history row matching symbol and lower market date",
        artifact_timestamp_utc=execution.isoformat(),
        observation_timestamp=str(rows.iloc[0].get("lower_date")),
    )
    return rows.iloc[[0]].copy(), evidence, execution


def _row_for_symbol(frame: pd.DataFrame | None, symbol: str) -> pd.Series | None:
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return None
    rows = frame.loc[frame["symbol"].astype(str).str.upper().eq(symbol)]
    return None if rows.empty else rows.iloc[-1]


def _observation_timestamp(row: pd.Series | None) -> Any:
    if row is None:
        return None
    if isinstance(row.name, (pd.Timestamp, datetime, date)):
        return row.name
    for field in ("date", "timestamp", "datetime"):
        if field in row.index:
            return row[field]
    return None


def load_indicator_params(
    config_path: Path, timeframe: str
) -> dict[str, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text())
    return config["indicators"]["stocks"][timeframe]


def recompute_indicators(
    bars: pd.DataFrame, timeframe: str, config_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    params = load_indicator_params(config_path, timeframe)
    clean = bars.copy()
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for indicator in INDICATORS:
        definition = params.get(indicator)
        if not definition:
            results[indicator] = np.nan
            errors[indicator] = "indicator is not configured for timeframe"
            continue
        func = INDICATOR_FUNCS[definition["id"]]
        kwargs = {key: value for key, value in definition.items() if key != "id"}
        try:
            series = func(clean, **kwargs)
            results[indicator] = series.iloc[-1] if not series.empty else np.nan
        except Exception as exc:  # evidence collection must retain partial results
            results[indicator] = np.nan
            errors[indicator] = f"{type(exc).__name__}: {exc}"
    return results, errors


def _safe_value(row: pd.Series | None, field: str) -> Any:
    return np.nan if row is None or field not in row.index else row[field]


def _update_inventory_time(
    evidence: S3Evidence, frame: pd.DataFrame | None, observation: Any = None
) -> S3Evidence:
    maximum = None
    if frame is not None and not frame.empty:
        converted = pd.to_datetime(frame.index, errors="coerce")
        valid = converted[~pd.isna(converted)]
        if len(valid):
            maximum = str(valid.max())
    values = asdict(evidence)
    values["observation_timestamp"] = None if observation is None else str(observation)
    values["maximum_bar_timestamp"] = maximum
    return S3Evidence(**values)


def local_output_dir(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"output directory must be under local root {root_resolved}")
    if str(path).startswith("s3://"):
        raise ValueError("S3 output paths are forbidden")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _classification(close_df: pd.DataFrame, indicator_df: pd.DataFrame) -> list[str]:
    classes: list[str] = []
    if (close_df["first_issue"] == "canonical_to_snapshot").any() or (
        indicator_df["recomputed_to_snapshot"] == "mismatch"
    ).any():
        classes.append("B. INDICATOR / SNAPSHOT ISSUE")
    if (close_df["first_issue"] == "snapshot_to_combo").any() or (
        indicator_df["snapshot_to_combo"] == "mismatch"
    ).any():
        classes.append("C. COMBO / ARTIFACT ISSUE")
    if (
        close_df["first_issue"]
        .isin(["canonical_to_snapshot_timestamp", "snapshot_to_combo_timestamp"])
        .any()
    ):
        classes.append("D. HISTORICAL ARTIFACT COHERENCE ISSUE")
    if close_df.astype(str).apply(
        lambda col: col.str.contains("unavailable")
    ).any().any() or (
        indicator_df.astype(str)
        .apply(lambda col: col.str.contains("unavailable"))
        .any()
        .any()
    ):
        classes.append("F. INSUFFICIENT HISTORICAL EVIDENCE")
    if not classes:
        classes.append("E. INTERNALLY CONSISTENT")
    return classes


def _markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free, human-readable table for the Markdown report."""
    return "```csv\n" + frame.to_csv(index=False).rstrip() + "\n```"


def run(args: argparse.Namespace, client: Any | None = None) -> Path:
    market_date = parse_market_date(args.market_date)
    stock = args.stock_symbol.upper()
    etfs = [args.primary_etf.upper(), args.secondary_etf.upper()]
    output = local_output_dir(Path(args.output_dir), Path.cwd())
    reader = ReadOnlyS3(args.bucket, args.prefix, client=client)
    inventory: list[S3Evidence] = []

    combo, combo_evidence, execution = find_historical_combo(reader, stock, market_date)
    inventory.append(combo_evidence)
    combo_row = None if combo is None else combo.iloc[0]

    if execution is None:
        # Still emit a complete unavailable-evidence bundle; never use latest data.
        execution = datetime.combine(market_date, time.max, tzinfo=timezone.utc)
        rolling_allowed = False
    else:
        rolling_allowed = True

    close_rows: list[dict[str, Any]] = []
    indicator_rows: list[dict[str, Any]] = []
    snapshot_rows: dict[str, pd.Series | None] = {}
    recomputed_by_tf: dict[str, dict[str, Any]] = {}

    for role, timeframe_name in ROLE_TIMEFRAMES.items():
        bars = snapshot = None
        if rolling_allowed:
            bars, bars_ev = reader.get_version_as_of(
                f"bars/stocks_{timeframe_name}/{stock}.parquet",
                execution,
                f"{stock}_{timeframe_name}_bars",
            )
            snapshot, snapshot_ev = reader.get_version_as_of(
                f"snapshot_stocks_{timeframe_name}.parquet",
                execution,
                f"stocks_{timeframe_name}_snapshot",
            )
        else:
            bars_ev = S3Evidence(
                f"{stock}_{timeframe_name}_bars",
                reader.key(f"bars/stocks_{timeframe_name}/{stock}.parquet"),
                None,
                None,
                "unavailable",
                "combo execution unavailable; rolling source not substituted",
            )
            snapshot_ev = S3Evidence(
                f"stocks_{timeframe_name}_snapshot",
                reader.key(f"snapshot_stocks_{timeframe_name}.parquet"),
                None,
                None,
                "unavailable",
                "combo execution unavailable; rolling source not substituted",
            )

        eligible = (
            filter_as_of(bars, market_date) if bars is not None else pd.DataFrame()
        )
        snapshot_row = _row_for_symbol(snapshot, stock)
        snapshot_rows[timeframe_name] = snapshot_row
        observation = _observation_timestamp(snapshot_row)
        inventory.extend(
            [
                _update_inventory_time(bars_ev, eligible),
                _update_inventory_time(snapshot_ev, snapshot, observation),
            ]
        )

        export = eligible.tail(HUMAN_BAR_ROWS).copy()
        export.insert(0, "timestamp", export.index.astype(str))
        export.insert(0, "timeframe", timeframe_name)
        export.insert(0, "symbol", stock)
        export.to_csv(
            output / f"{stock.lower()}_{timeframe_name}_bars.csv", index=False
        )

        recomputed: dict[str, Any] = {}
        errors: dict[str, str] = {}
        if not eligible.empty:
            recomputed, errors = recompute_indicators(
                eligible, timeframe_name, Path(args.indicator_config)
            )
        recomputed_by_tf[timeframe_name] = recomputed

        canonical_row = None if eligible.empty else eligible.iloc[-1]
        combo_timestamp = _safe_value(combo_row, f"{role}_date")
        close_rows.append(
            compare_close_lineage(
                timeframe_name,
                _safe_value(canonical_row, "close"),
                None if canonical_row is None else canonical_row.name,
                _safe_value(snapshot_row, "close"),
                observation,
                _safe_value(combo_row, f"{role}_close"),
                combo_timestamp,
                canonical_available=canonical_row is not None,
                snapshot_available=snapshot_row is not None,
                combo_available=combo_row is not None and f"{role}_close" in combo_row,
            )
        )
        for indicator in INDICATORS:
            row = compare_indicator_lineage(
                timeframe_name,
                indicator,
                recomputed.get(indicator, np.nan),
                _safe_value(snapshot_row, indicator),
                _safe_value(combo_row, f"{role}_{indicator}"),
                recomputed_available=indicator in recomputed
                and indicator not in errors,
                snapshot_available=snapshot_row is not None
                and indicator in snapshot_row,
                combo_available=combo_row is not None
                and f"{role}_{indicator}" in combo_row,
            )
            row["recompute_error"] = errors.get(indicator, "")
            indicator_rows.append(row)

    etf_rows: list[dict[str, Any]] = []
    for etf_symbol in etfs:
        etf_bars = etf_snapshot = etf_scores = None
        if rolling_allowed:
            etf_bars, bars_ev = reader.get_version_as_of(
                f"bars/stocks_weekly/{etf_symbol}.parquet",
                execution,
                f"{etf_symbol}_weekly_bars",
            )
            etf_snapshot, snap_ev = reader.get_version_as_of(
                "snapshot_etf_weekly.parquet",
                execution,
                "etf_weekly_snapshot",
            )
            etf_scores, score_ev = reader.get_version_as_of(
                "etf_trend_scores_weekly.parquet",
                execution,
                "etf_weekly_scores",
            )
        else:
            unavailable = lambda name, rel: S3Evidence(  # noqa: E731
                name,
                reader.key(rel),
                None,
                None,
                "unavailable",
                "combo execution unavailable; rolling source not substituted",
            )
            bars_ev = unavailable(
                f"{etf_symbol}_weekly_bars",
                f"bars/stocks_weekly/{etf_symbol}.parquet",
            )
            snap_ev = unavailable("etf_weekly_snapshot", "snapshot_etf_weekly.parquet")
            score_ev = unavailable(
                "etf_weekly_scores", "etf_trend_scores_weekly.parquet"
            )

        eligible = (
            filter_as_of(etf_bars, market_date)
            if etf_bars is not None
            else pd.DataFrame()
        )
        snapshot_row = _row_for_symbol(etf_snapshot, etf_symbol)
        observation = _observation_timestamp(snapshot_row)
        inventory.extend(
            [
                _update_inventory_time(bars_ev, eligible),
                _update_inventory_time(snap_ev, etf_snapshot, observation),
                _update_inventory_time(score_ev, etf_scores),
            ]
        )
        export = eligible.tail(HUMAN_BAR_ROWS).copy()
        export.insert(0, "timestamp", export.index.astype(str))
        export.insert(0, "timeframe", "weekly")
        export.insert(0, "symbol", etf_symbol)
        export.to_csv(output / f"{etf_symbol.lower()}_weekly_bars.csv", index=False)

        recomputed, errors = ({}, {})
        if not eligible.empty:
            recomputed, errors = recompute_indicators(
                eligible, "weekly", Path(args.indicator_config)
            )
        score_row = None
        if etf_scores is not None and not etf_scores.empty:
            if "etf_symbol" in etf_scores.columns:
                score_row = _row_for_symbol(
                    etf_scores.rename(columns={"etf_symbol": "symbol"}), etf_symbol
                )
            elif etf_symbol in etf_scores.index.astype(str):
                score_row = etf_scores.loc[etf_symbol]

        # Mirror the current scorer for audit without calling any loader/writer.
        wy = _safe_value(snapshot_row, "wyckoff_stage")
        mac = _safe_value(snapshot_row, "macdv_guard")
        wrong_volume_field = _safe_value(snapshot_row, "significant_volume")
        expected_long = (4.0 if wy == 2 else 0.0) + (2.0 if mac == 2 else 0.0)
        expected_short = (4.0 if wy == -2 else 0.0) + (2.0 if mac == -2 else 0.0)
        if wrong_volume_field == 1:
            expected_long += 1.0
            expected_short += 1.0
        etf_rows.append(
            {
                "symbol": etf_symbol,
                "timeframe": "weekly",
                "market_date": market_date.isoformat(),
                "canonical_max_timestamp": (
                    None if eligible.empty else str(eligible.index.max())
                ),
                "snapshot_timestamp": observation,
                "recomputed_wyckoff_stage": recomputed.get("wyckoff_stage", np.nan),
                "snapshot_wyckoff_stage": wy,
                "wyckoff_match": comparison(
                    recomputed.get("wyckoff_stage", np.nan),
                    wy,
                    available="wyckoff_stage" in recomputed
                    and "wyckoff_stage" not in errors
                    and snapshot_row is not None,
                ),
                "recomputed_macdv_guard": recomputed.get("macdv_guard", np.nan),
                "snapshot_macdv_guard": mac,
                "macdv_guard_match": comparison(
                    recomputed.get("macdv_guard", np.nan),
                    mac,
                    available="macdv_guard" in recomputed
                    and "macdv_guard" not in errors
                    and snapshot_row is not None,
                ),
                "snapshot_sig_vol_current_bar": _safe_value(
                    snapshot_row, "sig_vol_current_bar"
                ),
                "scorer_significant_volume_field": wrong_volume_field,
                "known_volume_schema_mismatch": "significant_volume"
                not in (snapshot_row.index if snapshot_row is not None else []),
                "score_recomputed_from_snapshot_long": expected_long,
                "score_recomputed_from_snapshot_short": expected_short,
                "cached_long_score": _safe_value(score_row, "etf_long_score"),
                "cached_short_score": _safe_value(score_row, "etf_short_score"),
                "cached_score_match": comparison(
                    (expected_long, expected_short),
                    (
                        _safe_value(score_row, "etf_long_score"),
                        _safe_value(score_row, "etf_short_score"),
                    ),
                    available=snapshot_row is not None and score_row is not None,
                ),
                "recompute_errors": json.dumps(errors, sort_keys=True),
            }
        )

    close_df = pd.DataFrame(close_rows)
    indicator_df = pd.DataFrame(indicator_rows)
    etf_df = pd.DataFrame(etf_rows)
    inventory_df = pd.DataFrame([asdict(item) for item in inventory]).drop_duplicates()
    close_df.to_csv(output / f"{stock.lower()}_close_lineage.csv", index=False)
    indicator_df.to_csv(output / f"{stock.lower()}_indicator_lineage.csv", index=False)
    etf_df.to_csv(output / "etf_lineage.csv", index=False)
    inventory_df.to_csv(output / "artifact_inventory.csv", index=False)
    if combo is not None:
        combo.to_csv(output / f"{stock.lower()}_historical_combo_row.csv", index=False)

    classifications = _classification(close_df, indicator_df)
    metadata = {
        "diagnostic": "scoring_data_lineage",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_date": market_date.isoformat(),
        "stock_symbol": stock,
        "primary_etf": etfs[0],
        "secondary_etf": etfs[1],
        "combo_id": COMBO_ID,
        "combo_execution_utc": combo_evidence.artifact_timestamp_utc,
        "as_of_policy": (
            "earliest immutable combo row matching lower market date; rolling inputs "
            "use latest S3 object version no later than that execution; bars after "
            "market_date are excluded; latest objects are never substituted"
        ),
        "classifications": classifications,
        "known_etf_volume_schema_mismatch": {
            "scorer_field": "significant_volume",
            "snapshot_field": "sig_vol_current_bar",
            "repaired_by_diagnostic": False,
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    report = [
        "# Scoring data-lineage diagnostic",
        "",
        f"- Market date: `{market_date}`",
        f"- Stock: `{stock}`",
        f"- Combo: `{COMBO_ID}`",
        f"- Combo execution UTC: `{combo_evidence.artifact_timestamp_utc}`",
        f"- ETFs: `{etfs[0]}`, `{etfs[1]}`",
        "- Mode: **read-only S3 evidence collection**",
        "",
        "## Evidence classification",
        "",
        *[f"- {item}" for item in classifications],
        "",
        "These are automated lineage classifications, not a production root-cause claim.",
        "",
        "## Close lineage",
        "",
        _markdown_table(close_df),
        "",
        "## Indicator lineage",
        "",
        _markdown_table(indicator_df),
        "",
        "## ETF lineage",
        "",
        _markdown_table(etf_df),
        "",
        "## Known ETF schema mismatch",
        "",
        "The current ETF scorer asks for `significant_volume`; configured snapshots expose "
        "`sig_vol_current_bar`. This diagnostic reports the mismatch and does not repair it.",
        "",
        "## Historical discipline",
        "",
        metadata["as_of_policy"],
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--market-date", required=True)
    parser.add_argument("--stock-symbol", required=True)
    parser.add_argument("--primary-etf", required=True)
    parser.add_argument("--secondary-etf", required=True)
    parser.add_argument("--output-dir", default="diagnostic_artifacts")
    parser.add_argument("--indicator-config", default="config/indicator_params.yaml")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
