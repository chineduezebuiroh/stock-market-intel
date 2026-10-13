"""Read persisted Stock EOD incident evidence; write reports locally only.

No runner, provider, publication, pointer, or storage write API is used. Interval
telemetry order is retained; acquisition order is derived only from unique UTC
fetch-start timestamps. Historical manifests have no pointer-bound manifest hash
here: artifact hashes are verified against the generation manifest, and both
manifest and telemetry byte hashes are recorded for audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import sys

import pandas as pd

from core import storage
from core.paths import DATA
from core import stock_eod_family as family
from diagnostics.validate_stock_eod_family_publication import DiagnosticFailure, _validate_generation

UNKNOWN = "UNKNOWN"
TIMEFRAMES = ("daily", "weekly", "monthly")
INTERVALS = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}
EXPECTED = {
    "prod-20261007T002633Z-c019d022": (2364, 2333, 2363, 2364, 2332),
    "prod-20261008T004729Z-ce8da38e": (2364, 2331, 2364, 2364, 2331),
    "prod-20261009T010208Z-57851ffe": (2364, 2343, 2364, 2364, 2343),
    "prod-20261009T235714Z-eb924d25": (2364, 2251, 2364, 2364, 2251),
    "prod-20261012T222708Z-399ff22b": (2330, 2298, 1613, 2330, 1613),
}
LATEST = list(EXPECTED)[-1]
FOCUS = ("COF", "PPLT", "QQQ", "SLV", "SPCX", "SPY", "TSLA", "V")
MIDNIGHT = pd.Timestamp("2026-10-13T00:00:00Z")
FIELDS = ("open", "high", "low", "close", "adj_close", "volume")
REQUIRED_FIELDS = ("open", "high", "low", "close", "volume")


def value(row, key):
    item = row.get(key, UNKNOWN)
    return UNKNOWN if item is None or (not isinstance(item, (dict, list)) and pd.isna(item)) else item


def provenance(row):
    item = value(row, "repair_provenance_json")
    try:
        parsed = json.loads(item) if item != UNKNOWN else {}
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def timestamp(item):
    # A timezone-less historical string cannot establish a UTC boundary.
    try:
        parsed = pd.Timestamp(item)
        return parsed.tz_convert("UTC") if parsed.tzinfo is not None else pd.NaT
    except (TypeError, ValueError):
        return pd.NaT


def validate_telemetry(frame, run_id):
    required = {"record_type", "family_run_id", "symbol", "interval"}
    if not required.issubset(frame.columns):
        raise DiagnosticFailure(f"{run_id}: telemetry missing identity columns {sorted(required - set(frame.columns))}")
    if frame.family_run_id.isna().any() or set(frame.family_run_id) != {run_id}:
        raise DiagnosticFailure(f"{run_id}: telemetry family identity mismatch")
    rows = frame.loc[frame.record_type == "interval"].copy().reset_index(drop=True)
    if rows.empty or rows.symbol.isna().any() or not set(rows.interval).issubset(INTERVALS):
        raise DiagnosticFailure(f"{run_id}: invalid interval telemetry")
    rows["symbol"] = rows.symbol.astype(str).str.strip().str.upper()
    if rows.symbol.eq("").any() or rows.duplicated(["symbol", "interval"]).any():
        raise DiagnosticFailure(f"{run_id}: duplicate/blank symbol interval telemetry")
    universe = set(rows.symbol)
    for interval, tf in INTERVALS.items():
        subset = rows.loc[rows.interval == interval]
        if set(subset.symbol) != universe:
            raise DiagnosticFailure(f"{run_id}: incomplete routed telemetry for {tf}")
        if "timeframe" in rows and (subset.timeframe != tf).any():
            raise DiagnosticFailure(f"{run_id}: interval/timeframe mismatch")
    summary = frame.loc[frame.record_type == "run_summary"]
    if len(summary) > 1:
        raise DiagnosticFailure(f"{run_id}: multiple run summaries")
    if not summary.empty and "routed_symbol_count" in summary:
        count = summary.iloc[0].routed_symbol_count
        if pd.notna(count) and count != len(universe):
            raise DiagnosticFailure(f"{run_id}: routed summary count mismatch")
    rows["timeframe"] = rows.interval.map(INTERVALS)
    positions = {s: i + 1 for i, s in enumerate(rows.symbol.drop_duplicates())}
    rows["telemetry_symbol_order"] = rows.symbol.map(positions)
    rows["acquisition_order"] = pd.Series(UNKNOWN, index=rows.index, dtype=object)
    for tf in TIMEFRAMES:
        subset = rows.loc[rows.timeframe == tf]
        times = subset.get("fetch_started_at", pd.Series(UNKNOWN, index=subset.index)).map(timestamp)
        if times.notna().all() and not times.duplicated().any():
            ordered = times.sort_values().index
            rows.loc[ordered, "acquisition_order"] = list(range(1, len(ordered) + 1))
    return rows.reset_index(drop=True), universe


def missing_sets(universe, frames):
    sets = {tf: set(frames[tf].symbol.astype(str).str.strip().str.upper()) for tf in TIMEFRAMES}
    if any(not symbols.issubset(universe) for symbols in sets.values()):
        raise DiagnosticFailure("immutable artifact contains symbols outside routed telemetry")
    intersection = set.intersection(*sets.values())
    return sets, {**{tf: universe - sets[tf] for tf in TIMEFRAMES}, "intersection": universe - intersection}, intersection


def reason(row):
    """One decisive category per missing observation; retain source reasons separately."""
    prov = provenance(row)
    text = " | ".join(str(value(row, k)) for k in (
        "acquisition_error", "quality_reason", "reconciliation_reason", "processing_reason"
    )) + " | " + str(prov.get("reason", ""))
    lower = text.lower()
    acquisition = value(row, "acquisition_status")
    if value(row, "processing_status") == "rejected":
        return "downstream_processing_rejection", text
    if value(row, "processing_status") == "accepted":
        # A repaired raw defect does not explain a later snapshot membership loss.
        return "unknown", text
    if acquisition in {"failed", "unavailable", "not_requested"}:
        return "acquisition_empty_unavailable_error", text
    if "stale" in lower or "current period" in lower or "current-period" in lower or "session mismatch" in lower:
        return "stale_session_current_period_failure", text
    if value(row, "reconciliation_status") == "conflict":
        return "reconciliation_conflict", text
    if "violates target high/low" in lower or "bounds" in lower:
        return "repair_bounds_failure", text
    if value(row, "reconciliation_status") in {"excluded", "failed", "unavailable"}:
        return "reconciliation_unavailable", text
    if "volume_negative" in lower or "negative volume" in lower:
        return "negative_invalid_volume", text
    for field in REQUIRED_FIELDS:
        item = value(row, "raw_" + field)
        if item != UNKNOWN:
            try:
                number = float(item)
                if field == "volume" and number < 0:
                    return "negative_invalid_volume", text
                if not math.isfinite(number):
                    return "nonfinite_required_fields", text
            except (TypeError, ValueError):
                return "invalid_required_fields", text
    if "nonfinite" in lower or "non-finite" in lower:
        return "nonfinite_required_fields", text
    if "invalid terminal required fields:" in lower or "target o/h/l/v is malformed" in lower:
        # _number() collapses null and nonfinite values to None in production telemetry.
        return "missing_or_nonfinite_required_fields", text
    if value(row, "finalization_status") in {"invalid", "excluded", "unavailable"} or value(row, "processing_status") == "excluded":
        return "finalized_observation_ineligible", text
    if any(value(row, k) != UNKNOWN for k in ("quality_reason", "processing_reason")) or prov.get("reason"):
        return "other", text
    return "unknown", text


def evidence_row(row, members):
    result = row.to_dict()
    # Every source column is retained, including JSON provenance and terminal numeric nulls.
    for key in (
        "fetch_started_at", "fetch_completed_at", "provider", "provider_version", "acquisition_status",
        "acquisition_error", "terminal_session", "session_evidence", "provider_metadata_json",
        "quality_status", "quality_reason", "reconciliation_status", "reconciliation_reason",
        "canonical_source_intervals", "reconciliation_tier", "finalization_status", "processing_status",
        "processing_reason", "repair_provenance_json", "raw_terminal_label", "raw_terminal_date",
    ):
        result.setdefault(key, UNKNOWN)
    for prefix in ("raw_", "finalized_"):
        for field in FIELDS:
            result.setdefault(prefix + field, UNKNOWN)
    status = value(row, "reconciliation_status")
    result["reconciliation_attempted"] = UNKNOWN if status == UNKNOWN else status != "not_needed"
    result["finalized_reason"] = provenance(row).get("reason", value(row, "quality_reason"))
    result["immutable_artifact_member"] = row.symbol in members[row.timeframe]
    result["reason_category"], result["supporting_reasons"] = reason(row)
    # Explicit presence flags disambiguate a persisted numeric NULL from an absent column.
    for field in FIELDS:
        result["raw_" + field + "_persisted"] = "raw_" + field in row.index
    shape_columns = ["raw_" + field for field in FIELDS]
    result["oct1_close_adj_null_ohlv_present"] = (
        UNKNOWN if not all(k in row.index for k in shape_columns) else
        all(pd.notna(row["raw_" + f]) for f in ("open", "high", "low", "volume"))
        and pd.isna(row.raw_close) and pd.isna(row.raw_adj_close)
    )
    return result


def stage_counts(run_id, rows, members):
    output = []
    metrics = {
        "acquisition_success": ("acquisition_status", {"success"}),
        "acquisition_failure_unavailable": ("acquisition_status", {"failed", "unavailable"}),
        "quality_valid": ("quality_status", {"valid"}),
        "quality_invalid": ("quality_status", {"invalid"}),
        "quality_unavailable": ("quality_status", {"unavailable"}),
        "reconciliation_repaired": ("finalization_status", {"repaired"}),
        "reconciliation_unrepaired_failed": ("reconciliation_status", {"excluded", "conflict", "failed", "unavailable"}),
        "finalized_valid": ("finalization_status", {"valid"}),
        "finalized_repaired": ("finalization_status", {"repaired"}),
        "finalized_ineligible_invalid": ("finalization_status", {"invalid", "excluded", "unavailable"}),
        "processing_accepted": ("processing_status", {"accepted"}),
        "processing_excluded": ("processing_status", {"excluded"}),
        "processing_rejected": ("processing_status", {"rejected"}),
        "processing_not_processed": ("processing_status", {"not_processed"}),
    }
    for tf in TIMEFRAMES:
        group = rows.loc[rows.timeframe == tf]
        for metric, column, choices in [(name, *spec) for name, spec in metrics.items()]:
            known = group[column].notna() & group[column].ne(UNKNOWN) if column in group else pd.Series(False, index=group.index)
            observed = int(group.loc[known, column].isin(choices).sum()) if column in group else 0
            output.append(dict(family_run_id=run_id, timeframe=tf, metric=metric,
                               count=observed if known.all() else UNKNOWN, observed_count=observed,
                               unknown_observations=int((~known).sum()), source=column))
        for metric, count, source in (
            ("routed_symbols", len(group), "complete interval telemetry"),
            ("immutable_artifact_membership", len(members[tf]), "verified generation"),
        ):
            output.append(dict(family_run_id=run_id, timeframe=tf, metric=metric, count=count,
                               observed_count=count, unknown_observations=0, source=source))
        statuses = group.get("acquisition_status", pd.Series(UNKNOWN, index=group.index))
        known = statuses.isin({"success", "failed", "unavailable", "not_requested"})
        attempted = int(statuses.isin({"success", "failed", "unavailable"}).sum())
        output.append(dict(family_run_id=run_id, timeframe=tf, metric="acquisition_attempted",
                           count=attempted if known.all() else UNKNOWN, observed_count=attempted,
                           unknown_observations=int((~known).sum()), source="acquisition_status (not repeated family request counters)"))
        statuses = group.get("reconciliation_status", pd.Series(UNKNOWN, index=group.index))
        known = statuses.notna() & statuses.ne(UNKNOWN)
        observed = int((known & statuses.ne("not_needed")).sum())
        output.append(dict(family_run_id=run_id, timeframe=tf, metric="reconciliation_attempted",
                           count=observed if known.all() else UNKNOWN, observed_count=observed,
                           unknown_observations=int((~known).sum()), source="persisted reconciliation result, not eligibility flag"))
    return output


def boundary_analysis(run_id, rows, missing):
    weekly = rows.loc[rows.timeframe == "weekly"]
    output = []
    for basis in ("fetch_started_at", "fetch_completed_at"):
        counts = Counter()
        for _, row in weekly.iterrows():
            ts = timestamp(value(row, basis))
            period = "unknown" if pd.isna(ts) else "before" if ts < MIDNIGHT else "at_or_after"
            counts[("missing" if row.symbol in missing else "present", period)] += 1
        for membership in ("missing", "present"):
            for period in ("before", "at_or_after", "unknown"):
                output.append(dict(family_run_id=run_id, timeframe="weekly", timestamp_basis=basis,
                                   boundary_utc=MIDNIGHT.isoformat(), membership=membership,
                                   period=period, count=counts[(membership, period)]))
    started = {(r["membership"], r["period"]): r["count"] for r in output if r["timestamp_basis"] == "fetch_started_at"}
    unknown = started[("missing", "unknown")] + started[("present", "unknown")]
    strict = bool(missing) and not unknown and not started[("missing", "before")] and not started[("present", "at_or_after")]
    counterexample = bool(started[("missing", "before")] or started[("present", "at_or_after")])
    verdict = "REJECTED" if counterexample else "UNRESOLVED" if unknown else "CONFIRMED" if strict else "REJECTED"
    # CONFIRMED means a perfect observed membership split, never provider causality.
    orders = weekly.acquisition_order
    suffix = UNKNOWN
    if orders.ne(UNKNOWN).all() and missing:
        lost_orders = sorted(int(r.acquisition_order) for _, r in weekly.iterrows() if r.symbol in missing)
        suffix = lost_orders == list(range(len(weekly) - len(missing) + 1, len(weekly) + 1))
    telemetry_orders = sorted(int(r.telemetry_symbol_order) for _, r in weekly.iterrows() if r.symbol in missing)
    telemetry_suffix = bool(missing) and telemetry_orders == list(range(len(weekly) - len(missing) + 1, len(weekly) + 1))
    return output, dict(verdict=verdict, acquisition_suffix=suffix, telemetry_suffix=telemetry_suffix,
                        mixed_losses=bool(started[("missing", "before")] and started[("missing", "at_or_after")]),
                        successful_after_midnight=started[("present", "at_or_after")], unknown_timestamps=unknown)


def overlaps(sets):
    output = []
    for (previous, a), (current, b) in zip(list(sets.items()), list(sets.items())[1:]):
        union = a | b
        output.append(dict(family_run_id=current, previous_family_run_id=previous, timeframe="daily",
                           previous_missing=len(a), current_missing=len(b), overlap=len(a & b),
                           union=len(union), jaccard=len(a & b) / len(union) if union else 1.0,
                           recurring_symbols=json.dumps(sorted(a & b))))
    return output


def table(frame, columns=None):
    if frame.empty:
        return "No rows."
    frame = frame[columns] if columns else frame
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join("| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))


def summarize(reports, gaps, boundary):
    coverage, detail = reports["family_coverage"], reports["focus_symbol_longitudinal"]
    latest_partition = reports["missing_reason_partition"]
    latest_partition = latest_partition.loc[(latest_partition.family_run_id == LATEST) & (latest_partition.timeframe == "weekly")]
    cof = detail.loc[(detail.symbol == "COF") & (detail.family_run_id == list(EXPECTED)[1])]
    fields = ["family_run_id", "symbol", "timeframe", "acquisition_status", "quality_status", "quality_reason", "reconciliation_status", "reconciliation_reason", "finalization_status", "processing_status", "immutable_artifact_member"]
    latest_focus = detail.loc[(detail.family_run_id == LATEST) & (detail.timeframe == "weekly")]
    losses = reports["family_missing_symbols"]
    daily = losses.loc[losses.timeframe == "daily"].reindex(columns=[*losses.columns, *([] if "oct1_close_adj_null_ohlv_present" in losses else ["oct1_close_adj_null_ohlv_present"])])
    distributions = daily.groupby(["family_run_id", "reason_category"]).size().reset_index(name="count")
    shape = losses.loc[losses.timeframe.isin(TIMEFRAMES)].reindex(columns=["family_run_id", "timeframe", "oct1_close_adj_null_ohlv_present"]).groupby(["family_run_id", "timeframe"]).oct1_close_adj_null_ohlv_present.value_counts(dropna=False).rename("count").reset_index()
    btext = json.dumps(boundary, sort_keys=True)
    return "\n\n".join([
        "# Stock EOD timeframe coverage incident diagnostic",
        "## A. Evidence integrity / generation verification\nAll selected manifests and artifact bytes passed schema, SHA-256, byte-size, row-count, symbol-count and retained-log assertions. Telemetry identities and complete unique D/W/M rows passed. Historical manifest authenticity is not independently pointer-bound; hashes recorded in family_coverage.csv. Only persisted evidence was read; no provider calls or S3 writes.",
        "## B. Family coverage timeline\n" + table(coverage, ["family_run_id", "routed_symbols", "daily", "weekly", "monthly", "intersection"]),
        "## C. Oct. 8 COF causal chain\n" + table(cof, fields) + "\nRead raw and finalized fields and provenance in focus_symbol_longitudinal.csv. Missing fields are not reconstructed from current providers. UNKNOWN prevents a complete upstream causal conclusion.",
        "## D. Oct. 13 weekly-collapse causal chain\n" + table(latest_focus, fields) + "\nEvery missing weekly observation, surviving D/M membership, persisted reasons, timestamps and order are in oct13_weekly_missing_detail.csv. These are observed stage outcomes, not proof of a provider-internal root cause.",
        "## E. Complete weekly reason partition\n" + table(latest_partition) + "\nCategories are mutually exclusive. Supporting raw reason combinations remain in the partition and per-symbol CSVs.",
        "## F. Midnight-UTC hypothesis\n" + btext + "\nCONFIRMED denotes an exact split of weekly membership by acquisition start around 2026-10-13 00:00:00 UTC; it does not prove midnight caused invalid data. REJECTED means a counterexample to that strict split; weaker correlation can remain. UNRESOLVED means missing/ambiguous timestamps. Completion-time counts are reported separately. Telemetry symbol order is persisted runner enumeration, not an explicitly stored processing position. Acquisition order requires all unique UTC timestamps.",
        "## G. Recurring daily-loss mechanism\n" + table(reports["daily_missing_overlap"]) + "\nPersisted daily loss categories:\n" + table(distributions) + "\n" + daily_findings(daily, reports["daily_loss_timing"]) + "\nRecurring-symbol reason stability is in daily_reason_recurrence.csv; time/order distributions are in daily_loss_timing.csv. No common upstream root cause is inferred from overlap alone.",
        "## H. Relationship to Oct. 1 monthly collapse\nHistorical comparison signature: O/H/L/V present, Close NULL, Adj Close NULL. Missing-observation signature counts:\n" + table(shape) + "\nMatching normalized terminal fields establishes a shared missing-field pattern only; SAME ROOT CAUSE is not established by that alone. Null and nonfinite provider values may have been collapsed by production telemetry.",
        "## I. Remaining evidence gaps\n" + "\n".join("- " + gap for gap in sorted(set(gaps))),
        "## J. Next investigation only\nWhere persisted outcomes leave upstream causality unresolved, inspect retained raw payload/version evidence and the provider normalization path at historical dependency versions. Do not reconstruct history with live calls. No production fix or semantic coverage threshold is proposed.",
    ]) + "\n"


def daily_analysis(records, acquired_rows=None):
    """Expose recurrence, reason changes and order/time clustering without guessing causes."""
    rows = pd.DataFrame(records)
    if rows.empty:
        return [], []
    rows = rows.loc[rows.timeframe == "daily"]
    recurrence, timing = [], []
    for symbol, group in rows.groupby("symbol"):
        if len(group) < 2:
            continue
        signatures = sorted(set(group.supporting_reasons))
        categories = sorted(set(group.reason_category))
        recurrence.append(dict(family_run_id="MULTIPLE", timeframe="daily", symbol=symbol,
            missing_family_count=len(group), family_run_ids=json.dumps(group.family_run_id.tolist()),
            reason_categories=json.dumps(categories), raw_reason_signatures=json.dumps(signatures),
            reason_stability="UNKNOWN" if "unknown" in categories else "STABLE" if len(signatures) == 1 else "VARIES"))
    for run_id, group in rows.groupby("family_run_id", sort=False):
        counts = Counter()
        population = Counter()
        def time_bin(row):
            ts = timestamp(value(row, "fetch_started_at"))
            hour = UNKNOWN if pd.isna(ts) else ts.floor("h").isoformat()
            order = value(row, "acquisition_order")
            bucket = UNKNOWN if order == UNKNOWN else f"{(int(order) - 1) // 100 * 100 + 1}-{(int(order) - 1) // 100 * 100 + 100}"
            return hour, bucket
        for _, row in group.iterrows():
            hour, bucket = time_bin(row)
            counts[(hour, bucket, row.reason_category)] += 1
        if acquired_rows is not None:
            for _, row in acquired_rows[run_id].iterrows():
                population[time_bin(row)] += 1
        for (hour, bucket, category), count in sorted(counts.items()):
            denominator = population[(hour, bucket)] if acquired_rows is not None else UNKNOWN
            timing.append(dict(family_run_id=run_id, timeframe="daily", acquisition_hour_utc=hour,
                               acquisition_order_bucket=bucket, reason_category=category, count=count,
                               routed_observations_in_bin=denominator,
                               missing_fraction_in_bin=count / denominator if denominator != UNKNOWN else UNKNOWN))
    return recurrence, timing


def daily_findings(daily, timing):
    oct8 = daily.loc[daily.family_run_id == list(EXPECTED)[1]]
    cof = oct8.loc[oct8.symbol == "COF"]
    text = "COF's Oct. 8 daily mechanism is UNKNOWN."
    if not cof.empty and not oct8.empty:
        dominant = oct8.reason_category.value_counts()
        category = cof.iloc[0].reason_category
        leaders = dominant.loc[dominant == dominant.max()].index.tolist()
        text = f"COF Oct. 8 category={category}; dominant daily-loss categories={leaders}; matches dominant category={category in leaders}. This compares persisted exclusion mechanisms, not upstream causes."
    if not timing.empty:
        text += " Observed daily missing populations are grouped by UTC acquisition hour and 100-symbol acquisition-order bins; the CSV preserves all bins rather than assigning a speculative clustering threshold."
    return text


def run_diagnostic(*, output_dir, data_root=DATA, run_ids=None, expectations=None):
    run_ids = list(EXPECTED) if run_ids is None else list(run_ids)
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise DiagnosticFailure("family list must be nonempty and unique")
    expectations = EXPECTED if expectations is None else expectations
    if str(output_dir).startswith("s3:"):
        raise DiagnosticFailure("output must be a local filesystem directory")
    output_dir, data_root = Path(output_dir).resolve(), Path(data_root).resolve()
    if output_dir == data_root or data_root in output_dir.parents:
        raise DiagnosticFailure("local output directory must be outside the production data root")
    reports = {name: [] for name in (
        "family_coverage", "family_stage_counts", "family_missing_symbols", "missing_reason_partition",
        "focus_symbol_longitudinal", "oct13_weekly_missing_detail", "daily_missing_overlap", "midnight_boundary_analysis",
        "daily_reason_recurrence", "daily_loss_timing",
    )}
    daily_sets, daily_rows, gaps, boundary = {}, {}, [], {"verdict": "UNRESOLVED", "reason": "latest family not selected"}
    for run_id in run_ids:
        family.validate_family_run_id(run_id)
        _, frames = _validate_generation(run_id, data_root=data_root, label=run_id)
        manifest_bytes = storage.read_bytes(family.manifest_path(run_id, data_root))
        telemetry_bytes = storage.read_bytes(data_root / "_health" / "stocks_eod_family" / run_id / "manifest.parquet")
        telemetry = pd.read_parquet(BytesIO(telemetry_bytes))
        rows, universe = validate_telemetry(telemetry, run_id)
        daily_rows[run_id] = rows.loc[rows.timeframe == "daily"]
        members, missing, intersection = missing_sets(universe, frames)
        actual = (len(universe), *(len(members[tf]) for tf in TIMEFRAMES), len(intersection))
        if run_id in expectations and actual != tuple(expectations[run_id]):
            raise DiagnosticFailure(f"{run_id}: retained-log coverage mismatch expected={expectations[run_id]} actual={actual}")
        reports["family_coverage"].append(dict(family_run_id=run_id, timeframe="D/W/M", routed_symbols=actual[0],
            daily=actual[1], weekly=actual[2], monthly=actual[3], intersection=actual[4],
            manifest_sha256=sha256(manifest_bytes).hexdigest(), telemetry_sha256=sha256(telemetry_bytes).hexdigest(),
            verification="PASS", retained_log_assertion="PASS" if run_id in expectations else "UNKNOWN"))
        reports["family_stage_counts"].extend(stage_counts(run_id, rows, members))
        evidence = [evidence_row(row, members) for _, row in rows.iterrows()]
        reports["focus_symbol_longitudinal"].extend(r for r in evidence if r["symbol"] in FOCUS)
        for tf in TIMEFRAMES:
            selected = [r for r in evidence if r["timeframe"] == tf and r["symbol"] in missing[tf]]
            reports["family_missing_symbols"].extend(selected)
            counts = Counter((r["reason_category"], r["supporting_reasons"]) for r in selected)
            for (category, support), count in sorted(counts.items()):
                reports["missing_reason_partition"].append(dict(family_run_id=run_id, timeframe=tf,
                    reason_category=category, supporting_reasons=support, count=count, total_missing=len(missing[tf])))
            if sum(counts.values()) != len(missing[tf]):
                raise DiagnosticFailure(f"{run_id}: {tf} reason partition does not reconcile")
        for symbol in sorted(missing["intersection"]):
            reports["family_missing_symbols"].append(dict(family_run_id=run_id, timeframe="intersection", symbol=symbol,
                reason_category="missing_required_timeframes", supporting_reasons=",".join(tf for tf in TIMEFRAMES if symbol in missing[tf])))
        daily_sets[run_id] = missing["daily"]
        if run_id == LATEST:
            for r in evidence:
                if r["timeframe"] == "weekly" and r["symbol"] in missing["weekly"]:
                    reports["oct13_weekly_missing_detail"].append({**r, "daily_survived": r["symbol"] in members["daily"], "monthly_survived": r["symbol"] in members["monthly"]})
            counts, boundary = boundary_analysis(run_id, rows, missing["weekly"])
            reports["midnight_boundary_analysis"].extend(counts)
        gaps.append(f"{run_id}: raw terminal interval label/date and explicit processing position are not present in the current telemetry contract; terminal_session is separate derived evidence.")
        unknown_metrics = [r["metric"] + "/" + r["timeframe"] for r in reports["family_stage_counts"] if r["family_run_id"] == run_id and r["count"] == UNKNOWN]
        if unknown_metrics:
            gaps.append(f"{run_id}: incomplete telemetry stage fields: {', '.join(unknown_metrics)}")
    reports["daily_missing_overlap"] = overlaps(daily_sets)
    reports["daily_reason_recurrence"], reports["daily_loss_timing"] = daily_analysis(reports["family_missing_symbols"], daily_rows)
    # Validate every family before writing any success reports; writes use Path/pandas only.
    minimum = {
        "family_coverage": ["family_run_id", "timeframe"],
        "family_stage_counts": ["family_run_id", "timeframe", "metric", "count"],
        "family_missing_symbols": ["family_run_id", "timeframe", "symbol", "reason_category", "supporting_reasons", "oct1_close_adj_null_ohlv_present"],
        "missing_reason_partition": ["family_run_id", "timeframe", "reason_category", "supporting_reasons", "count", "total_missing"],
        "focus_symbol_longitudinal": ["family_run_id", "timeframe", "symbol"],
        "oct13_weekly_missing_detail": ["family_run_id", "timeframe", "symbol"],
        "daily_missing_overlap": ["family_run_id", "previous_family_run_id", "timeframe"],
        "midnight_boundary_analysis": ["family_run_id", "timeframe", "timestamp_basis", "membership", "period", "count"],
        "daily_reason_recurrence": ["family_run_id", "timeframe", "symbol", "reason_stability"],
        "daily_loss_timing": ["family_run_id", "timeframe", "acquisition_hour_utc", "acquisition_order_bucket", "reason_category", "count"],
    }
    frames_out = {name: pd.DataFrame(records) if records else pd.DataFrame(columns=minimum[name]) for name, records in reports.items()}
    summary = summarize(frames_out, gaps, boundary)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames_out.items():
        frame.to_csv(output_dir / (name + ".csv"), index=False, na_rep="NULL")
    (output_dir / "incident_summary.md").write_text(summary)
    return frames_out, boundary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run_diagnostic(output_dir=args.output_dir)
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[PASS] Reports written locally to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
