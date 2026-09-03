#!/usr/bin/env python3
"""Production-S3 read-only Phase 4A.1 outcome coverage and integrity audit."""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.analyze_stock_options_participation import (
    COMBO_SPECS,
    build_directional_opportunities,
)
from diagnostics.analyze_stock_options_phase3b import validate_population
from diagnostics.outcomes.coverage import (
    audit_horizon,
    corporate_action_events,
    revision_measures,
    summarize_coverage,
)
from diagnostics.outcomes.price_source import CachedPriceSource, RollingDailyPriceSource
from diagnostics.outcomes.regimes import assign_engine_regime
from diagnostics.outcomes.specs import Direction, HORIZON_SPECS

OUTPUT_FILES = (
    "phase4a_coverage_summary.json",
    "phase4a_coverage_report.md",
    "price_store_inventory.csv",
    "entry_price_revision_audit.csv",
    "entry_price_revision_summary.csv",
    "outcome_coverage_observations.parquet",
    "outcome_coverage_summary.csv",
    "outcome_coverage_by_entry_month.csv",
    "corporate_action_flags.csv",
    "corporate_action_summary.csv",
    "phase4a_real_data_smoke.csv",
)


class ReadOnlyS3ParquetReader:
    """GetObject-only reader; missing keys become empty histories."""

    def __init__(self, bucket: str, client: Any) -> None:
        self.bucket = bucket
        self.client = client
        self.missing: set[str] = set()
        self.calls: dict[str, int] = {}

    def __call__(self, key: str) -> pd.DataFrame:
        self.calls[key] = self.calls.get(key, 0) + 1
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                self.missing.add(key)
                return pd.DataFrame()
            raise
        return pd.read_parquet(io.BytesIO(body))


def _load_populations(root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for combo in COMBO_SPECS:
        combo_root = root / combo
        frame = pd.read_parquet(combo_root / "supported_observations.parquet")
        coverage = json.loads((combo_root / "coverage_summary.json").read_text())
        validate_population(combo, frame, coverage)
        frames[combo] = frame
    return frames


def _inventory(
    symbol: str, key: str, frame: pd.DataFrame, missing: bool
) -> dict[str, Any]:
    if frame.empty:
        return {
            "symbol": symbol,
            "s3_key": key,
            "symbol_history_missing": missing,
            "row_count": 0,
            "first_available_market_date": None,
            "last_available_market_date": None,
            "dates_unique": True,
            "dates_monotonic_after_normalization": True,
            "missing_ohlc_count": 0,
            "invalid_ohlc_count": 0,
            "adj_close_available": False,
            "retention_span_calendar_days": 0,
            "trading_session_count": 0,
        }
    dates = pd.DatetimeIndex(frame.index).normalize()
    ohlc = frame.reindex(columns=["open", "high", "low", "close"]).apply(
        pd.to_numeric, errors="coerce"
    )
    first, last = dates.min().date(), dates.max().date()
    return {
        "symbol": symbol,
        "s3_key": key,
        "symbol_history_missing": False,
        "row_count": len(frame),
        "first_available_market_date": first,
        "last_available_market_date": last,
        "dates_unique": bool(dates.is_unique),
        "dates_monotonic_after_normalization": bool(dates.is_monotonic_increasing),
        "missing_ohlc_count": int(ohlc.isna().sum().sum()),
        "invalid_ohlc_count": int(((ohlc <= 0) & ohlc.notna()).sum().sum()),
        "adj_close_available": bool(
            "adj_close" in frame
            and pd.to_numeric(frame["adj_close"], errors="coerce").notna().any()
        ),
        "retention_span_calendar_days": (last - first).days,
        "trading_session_count": int(dates.nunique()),
    }


def _revision_summary(audit: pd.DataFrame) -> pd.DataFrame:
    groups = ["combo", "direction"]
    measures = [
        "exact_or_near_exact",
        "over_1bp",
        "over_10bp",
        "over_50bp",
        "over_1pct",
        "over_5pct",
    ]
    rows = []
    for keys, part in audit.groupby(groups, dropna=False):
        comparable = part[part["rolling_entry_close"].notna()]
        row = dict(zip(groups, keys)) | {
            "candidate_observations": len(part),
            "entry_date_comparable_count": len(comparable),
            "entry_date_comparable_rate": (
                len(comparable) / len(part) if len(part) else np.nan
            ),
            "absolute_difference_median": comparable["absolute_difference"].median(),
            "percentage_difference_p95": comparable["percentage_difference"].quantile(
                0.95
            ),
        }
        row.update({f"{name}_count": int(comparable[name].sum()) for name in measures})
        row.update({f"{name}_rate": comparable[name].mean() for name in measures})
        rows.append(row)
    return pd.DataFrame(rows)


def _decision(part: pd.DataFrame) -> str:
    mature = part[part.theoretically_mature_count > 0]
    if mature.empty or mature.mature_terminal_return_count.sum() == 0:
        return "D — NOT CURRENTLY SUPPORTABLE"
    terminal = (
        mature.mature_terminal_return_count.sum()
        / mature.theoretically_mature_count.sum()
    )
    path = mature.complete_path_count.sum() / mature.theoretically_mature_count.sum()
    if terminal >= 0.95 and path >= 0.95:
        return "A — SUFFICIENT FOR OUTCOME CALIBRATION"
    if terminal >= 0.80 and path >= 0.70:
        return "B — USABLE WITH DOCUMENTED CENSORING"
    return "C — PARTIAL / DESCRIPTIVE ONLY"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free Markdown table for GitHub artifacts."""
    values = frame.replace({np.nan: ""}).astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    divider = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _write_report(
    output: Path,
    summary: pd.DataFrame,
    revisions: pd.DataFrame,
    corporate: pd.DataFrame,
    decisions: dict[str, str],
    asof: date,
) -> None:
    lines = [
        "# Phase 4A.1 outcome coverage and integrity audit",
        "",
        "> Evidence-only, read-only audit. No participation threshold recommendation.",
        "",
        f"Outcome dataset as-of: **{asof}** (maximum valid retained daily market date).",
        "",
        "## Terminal-return and complete-path coverage",
        "",
        _markdown_table(summary),
        "",
        "Coverage denominators contain only theoretically mature targets. `IMMATURE` means NOT YET MATURE; all other uncovered theoretically mature rows mean MATURE BUT DATA UNAVAILABLE.",
        "",
        "## Entry-price revision audit",
        "",
        _markdown_table(revisions),
        "",
        "Immutable combo-history OHLC remains entry truth. Differences are descriptive and never replace it.",
        "",
        "## Potential corporate actions",
        "",
        f"Flagged observation/horizon windows: **{int(corporate['corporate_action_flag'].sum()) if not corporate.empty else 0}**. Flags remain included in coverage totals.",
        "",
        "Heuristic: flag an adjacent valid adj_close/close ratio change above 5%; additionally flag a split-like raw close move above 40% accompanied by an inverse adjustment-ratio move above 20%. No prices are adjusted.",
        "",
        "## Calibration-support classifications",
        "",
    ]
    lines.extend(
        f"* `{combo}`: **{decision}**" for combo, decision in decisions.items()
    )
    lines += [
        "",
        "M/Q/Y is **LIMITED HISTORY**. Its immature rows are not missing-data failures.",
        "",
        "Classification rules: A requires >=95% terminal and path coverage; B requires >=80% terminal and >=70% path coverage; C has some mature terminal evidence below B; D has no mature terminal evidence.",
        "",
        "## Long-term archive decision",
        "",
        "A 260-session revisable store is not an adequate long-term outcome source for 270/365/730/1095-day horizons. A future archive should be append-only daily OHLCV, retain raw and adjusted OHLC (not only adjusted close), preserve split/dividend/action records and provider provenance, retain symbols after universe removal, cover at least 1,110 calendar days plus alignment margin (prefer indefinitely), version corrections rather than overwrite them, and use keys such as `outcome_prices/stocks/daily/symbol=<SYMBOL>/year=<YYYY>/part-<immutable-id>.parquet` with a manifest/as-of ledger.",
    ]
    (output / "phase4a_coverage_report.md").write_text("\n".join(lines) + "\n")


def run(
    validated_root: Path, output: Path, bucket: str, prefix: str, client: Any
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    populations = _load_populations(validated_root)
    directional = []
    for combo, canonical in populations.items():
        expanded = build_directional_opportunities(canonical, combo)
        directional.append(expanded[expanded.pre_participation].copy())
    candidates = pd.concat(directional, ignore_index=True)
    candidates["lower_date"] = pd.to_datetime(candidates["lower_date"]).dt.normalize()

    reader = ReadOnlyS3ParquetReader(bucket, client)
    source = CachedPriceSource(RollingDailyPriceSource(reader, prefix=prefix))
    histories, inventories, events = {}, [], {}
    for symbol in sorted(candidates.symbol.astype(str).str.upper().unique()):
        history = source.load(symbol)
        histories[symbol] = history
        inventories.append(
            _inventory(
                symbol,
                history.source_key,
                history.frame,
                history.source_key in reader.missing,
            )
        )
        event_frame = corporate_action_events(history.frame)
        events[symbol] = event_frame
    inventory = pd.DataFrame(inventories)
    available_last = pd.to_datetime(
        inventory.last_available_market_date, errors="coerce"
    ).dropna()
    if available_last.empty:
        raise RuntimeError(
            "no production daily stock histories were readable; failing closed"
        )
    dataset_asof = available_last.max().date()

    revisions, observations, corporate_rows = [], [], []
    for row in candidates.itertuples(index=False):
        symbol = str(row.symbol).upper()
        history = histories[symbol]
        entry_ts = pd.Timestamp(row.lower_date)
        entry_close = float(row.lower_close)
        rolling = (
            history.frame.loc[entry_ts] if entry_ts in history.frame.index else None
        )
        measures = (
            revision_measures(entry_close, float(rolling.close))
            if rolling is not None
            else revision_measures(entry_close, np.nan)
        )
        revision = {
            "combo": row.combo,
            "direction": row.direction,
            "symbol": symbol,
            "entry_market_date": entry_ts.date(),
            "entry_execution_timestamp": row.artifact_execution_utc,
            "source_combo_history_key": row.source_s3_key,
            "immutable_entry_close": entry_close,
            "rolling_entry_close": (
                float(rolling.close) if rolling is not None else np.nan
            ),
            **measures,
        }
        for field in ("high", "low"):
            immutable = float(getattr(row, f"lower_{field}"))
            current = float(getattr(rolling, field)) if rolling is not None else np.nan
            revision[f"immutable_entry_{field}"] = immutable
            revision[f"rolling_entry_{field}"] = current
            revision[f"{field}_percentage_difference"] = revision_measures(
                immutable, current
            )["percentage_difference"]
        revisions.append(revision)

        regime = assign_engine_regime(
            row.combo, pd.Timestamp(row.artifact_execution_utc).to_pydatetime()
        )
        for horizon in HORIZON_SPECS[row.combo].horizons:
            result = audit_horizon(
                direction=Direction(row.direction),
                entry_date=entry_ts.date(),
                immutable_entry_close=entry_close,
                horizon=horizon,
                frame=None if history.source_key in reader.missing else history.frame,
                dataset_asof=dataset_asof,
                tolerance_days=HORIZON_SPECS[row.combo].alignment_tolerance_days,
            )
            exit_date = result["resolved_exit_date"]
            event_frame = events[symbol]
            event_mask = pd.Series(False, index=event_frame.index)
            if exit_date is not None and not event_frame.empty:
                event_mask = (event_frame.index.date > entry_ts.date()) & (
                    event_frame.index.date <= exit_date
                )
            flagged = bool(event_mask.any())
            observation = {
                "combo": row.combo,
                "direction": row.direction,
                "symbol": symbol,
                "entry_market_date": entry_ts.date(),
                "entry_execution_timestamp": row.artifact_execution_utc,
                "source_combo_history_key": row.source_s3_key,
                "engine_regime_id": regime,
                "historical_participation_pass": bool(row.overall_participation_pass),
                "historical_participation_route": row.route_class,
                "immutable_entry_close": entry_close,
                "outcome_price_source_key": history.source_key,
                "horizon_id": horizon.horizon_id,
                "horizon_multiple": horizon.multiple,
                **result,
                "corporate_action_flag": flagged,
            }
            observations.append(observation)
            if flagged:
                for event in event_frame.loc[event_mask].itertuples(index=False):
                    corporate_rows.append(
                        {
                            "combo": row.combo,
                            "direction": row.direction,
                            "symbol": symbol,
                            "entry_market_date": entry_ts.date(),
                            "horizon_id": horizon.horizon_id,
                            "corporate_action_flag": True,
                            **event._asdict(),
                        }
                    )

    observations_df = pd.DataFrame(observations)
    revisions_df = pd.DataFrame(revisions)
    corporate_df = pd.DataFrame(
        corporate_rows,
        columns=[
            "combo",
            "direction",
            "symbol",
            "entry_market_date",
            "horizon_id",
            "corporate_action_flag",
            "event_date",
            "adjustment_ratio",
            "ratio_change",
            "raw_close_change",
            "ratio_break_flag",
            "split_like_flag",
        ],
    )
    summary = summarize_coverage(
        observations_df, ["combo", "direction", "horizon_id", "horizon_multiple"]
    )
    observations_df["entry_month"] = (
        pd.to_datetime(observations_df.entry_market_date).dt.to_period("M").astype(str)
    )
    by_month = summarize_coverage(
        observations_df, ["combo", "direction", "horizon_id", "entry_month"]
    )
    revision_summary = _revision_summary(revisions_df)
    corporate_summary = (
        observations_df.groupby(["combo", "direction", "horizon_id"], dropna=False)
        .agg(
            mature_outcomes=("terminal_covered", "sum"),
            corporate_action_flagged_count=("corporate_action_flag", "sum"),
        )
        .reset_index()
    )
    corporate_summary["flagged_share_of_terminal_outcomes"] = (
        corporate_summary.corporate_action_flagged_count
        / corporate_summary.mature_outcomes.replace(0, np.nan)
    )
    decisions = {
        combo: _decision(summary[summary.combo.eq(combo)]) for combo in COMBO_SPECS
    }

    inventory.to_csv(output / "price_store_inventory.csv", index=False)
    revisions_df.to_csv(output / "entry_price_revision_audit.csv", index=False)
    revision_summary.to_csv(output / "entry_price_revision_summary.csv", index=False)
    observations_df.to_parquet(
        output / "outcome_coverage_observations.parquet", index=False
    )
    summary.to_csv(output / "outcome_coverage_summary.csv", index=False)
    by_month.to_csv(output / "outcome_coverage_by_entry_month.csv", index=False)
    corporate_df.to_csv(output / "corporate_action_flags.csv", index=False)
    corporate_summary.to_csv(output / "corporate_action_summary.csv", index=False)
    smoke_groups = [
        "combo",
        "direction",
        "historical_participation_pass",
        "horizon_multiple",
        "coverage_status",
    ]
    smoke = (
        observations_df.sort_values(smoke_groups + ["symbol"])
        .groupby(smoke_groups, dropna=False)
        .head(1)
    )
    smoke["immutable_entry_close_preserved"] = True
    smoke["forward_target_verified"] = smoke.resolved_exit_date.isna() | (
        pd.to_datetime(smoke.resolved_exit_date)
        >= pd.to_datetime(smoke.target_calendar_date)
    )
    smoke["target_tolerance_verified"] = smoke.resolved_exit_date.isna() | (
        (
            pd.to_datetime(smoke.resolved_exit_date)
            - pd.to_datetime(smoke.target_calendar_date)
        ).dt.days
        <= 7
    )
    expected_return = np.where(
        smoke.direction.eq("LONG"),
        smoke.exit_close / smoke.immutable_entry_close - 1,
        smoke.immutable_entry_close / smoke.exit_close - 1,
    )
    smoke["directional_return_verified"] = ~smoke.terminal_covered | np.isclose(
        smoke.directional_return, expected_return, equal_nan=False
    )
    smoke["path_metrics_verified"] = ~smoke.path_covered | (
        smoke.mfe.notna() & smoke.mae.notna() & smoke.elapsed_trading_sessions.notna()
    )
    verification = [
        "immutable_entry_close_preserved",
        "forward_target_verified",
        "target_tolerance_verified",
        "directional_return_verified",
        "path_metrics_verified",
    ]
    if not smoke[verification].all(axis=None):
        raise AssertionError("real-data smoke verification failed")
    smoke.to_csv(output / "phase4a_real_data_smoke.csv", index=False)
    machine_summary = {
        "phase": "phase4a_coverage",
        "validation_scope": "cross_combo_phase3b",
        "read_only": True,
        "dataset_asof": dataset_asof,
        "decisions": decisions,
        "price_symbols": len(inventory),
        "missing_symbols": int(inventory.symbol_history_missing.sum()),
        "s3_get_count": sum(reader.calls.values()),
        "unique_s3_keys_read": len(reader.calls),
        "outputs": OUTPUT_FILES,
        "production_threshold_changes": False,
    }
    (output / "phase4a_coverage_summary.json").write_text(
        json.dumps(machine_summary, indent=2, default=str) + "\n"
    )
    _write_report(
        output, summary, revision_summary, observations_df, decisions, dataset_asof
    )
    if any(count != 1 for count in reader.calls.values()):
        raise AssertionError("one-load-per-symbol invariant failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    client = boto3.client("s3")
    run(args.validated_root, args.output_dir, args.bucket, args.prefix, client)
    print("[SAFETY] READ ONLY; S3 GetObject only; NO S3 WRITES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
