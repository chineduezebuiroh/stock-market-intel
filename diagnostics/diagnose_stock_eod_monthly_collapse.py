"""Read-only diagnostic for the 2026-10-01 Stock EOD monthly collapse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

from core import storage
from core.paths import DATA
from core.stock_eod_family import artifact_path


FAMILY_RUN_ID = "prod-20261002T003510Z-85b4ac62"
PREVIOUS_FAMILY_RUN_ID = "prod-20261001T002500Z-c924d845"
MAX_SAMPLE_ROWS = 20

GROUP_COLUMNS = (
    "acquisition_status",
    "quality_status",
    "quality_reason",
    "close_is_null",
    "finalization_status",
    "processing_status",
    "processing_reason",
)
VALUE_COLUMNS = (
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_adj_close",
    "raw_volume",
)
SAMPLE_COLUMNS = (
    "symbol",
    "terminal_session",
    "session_evidence",
    "quality_status",
    "quality_reason",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_adj_close",
    "raw_volume",
    "finalization_status",
    "processing_status",
    "processing_reason",
)
REQUIRED_COLUMNS = {
    "record_type",
    "interval",
    "acquisition_error",
    *GROUP_COLUMNS,
    *VALUE_COLUMNS,
    *SAMPLE_COLUMNS,
}


class DiagnosticFailure(RuntimeError):
    """The persisted evidence cannot support the requested diagnostic."""


@dataclass(frozen=True)
class FailureCounts:
    null_raw_close_shape: int
    acquisition_failure: int
    other_required_field_failure: int
    post_acquisition_processing_failure: int


def filter_monthly_intervals(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate the manifest schema and return native monthly interval records."""
    missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise DiagnosticFailure(f"manifest missing required columns: {missing}")
    monthly = manifest.loc[
        manifest["record_type"].eq("interval") & manifest["interval"].eq("1mo")
    ].copy()
    if monthly.empty:
        raise DiagnosticFailure("manifest contains no monthly interval rows")
    return monthly


def null_raw_close_pattern(frame: pd.DataFrame) -> pd.Series:
    """Identify the precise malformed shape under investigation."""
    return (
        frame["raw_close"].isna()
        & frame["raw_adj_close"].notna()
        & frame[["raw_open", "raw_high", "raw_low", "raw_volume"]].notna().all(axis=1)
    )


def failure_counts(monthly: pd.DataFrame) -> FailureCounts:
    """Partition observed failed rows into deterministic diagnostic categories."""
    acquisition_failed = monthly["acquisition_status"].ne("success")
    close_shape = ~acquisition_failed & null_raw_close_pattern(monthly)
    quality_failed = monthly["quality_status"].ne("valid")
    other_required = ~acquisition_failed & ~close_shape & quality_failed
    finalized_eligible = monthly["finalization_status"].isin({"valid", "repaired"})
    post_acquisition = (
        ~acquisition_failed
        & ~quality_failed
        & (~finalized_eligible | monthly["processing_status"].ne("accepted"))
    )
    return FailureCounts(
        int(close_shape.sum()),
        int(acquisition_failed.sum()),
        int(other_required.sum()),
        int(post_acquisition.sum()),
    )


def classify_failure(counts: FailureCounts) -> str:
    """Classify only the failure modes directly represented by persisted fields."""
    observed = [
        (counts.null_raw_close_shape, "CONFIRMED_NULL_RAW_CLOSE_BOUNDARY_SHAPE"),
        (counts.acquisition_failure, "CONFIRMED_PROVIDER_ACQUISITION_FAILURE"),
        (counts.other_required_field_failure, "CONFIRMED_OTHER_REQUIRED_FIELD_FAILURE"),
        (
            counts.post_acquisition_processing_failure,
            "CONFIRMED_POST_ACQUISITION_PROCESSING_FAILURE",
        ),
    ]
    classifications = [name for count, name in observed if count > 0]
    if len(classifications) > 1:
        return "MIXED_FAILURE_MODES"
    if classifications:
        return classifications[0]
    return "INCONCLUSIVE"


def _print_group_counts(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    counts = frame.groupby(list(columns), dropna=False).size().rename("count").reset_index()
    print(counts.to_string(index=False))


def _print_manifest_analysis(monthly: pd.DataFrame) -> FailureCounts:
    print(f"[MONTHLY] total_interval_rows={len(monthly)}")
    for column in GROUP_COLUMNS:
        print(f"\n[GROUP_COUNTS] {column}")
        _print_group_counts(monthly, [column])

    print("\n[NULL_COUNTS]")
    for column in VALUE_COLUMNS:
        nulls = int(monthly[column].isna().sum())
        print(f"{column} null={nulls} non_null={len(monthly) - nulls}")

    malformed = null_raw_close_pattern(monthly)
    print(f"\n[PATTERN] null_raw_close_with_adj_close_and_ohlv={int(malformed.sum())}")

    print("\n[TERMINAL_EVIDENCE_COUNTS]")
    _print_group_counts(monthly, ["terminal_session", "session_evidence", "close_is_null"])

    acquisition_failures = monthly.loc[monthly["acquisition_status"].ne("success")]
    print(f"\n[ACQUISITION_FAILURES] count={len(acquisition_failures)}")
    if not acquisition_failures.empty:
        _print_group_counts(
            acquisition_failures,
            ["acquisition_status", "acquisition_error", "quality_reason"],
        )

    print("\n[FINALIZATION_PROCESSING_COUNTS]")
    _print_group_counts(monthly, ["finalization_status", "processing_status", "processing_reason"])

    print(f"\n[SUSPECTED_MALFORMED_SAMPLE] max_rows={MAX_SAMPLE_ROWS}")
    print(monthly.loc[malformed, list(SAMPLE_COLUMNS)].head(MAX_SAMPLE_ROWS).to_string(index=False))

    accepted = monthly.loc[monthly["processing_status"].eq("accepted")]
    print(f"\n[ACCEPTED_MONTHLY_ROWS] count={len(accepted)}")
    if len(accepted) <= MAX_SAMPLE_ROWS:
        print(accepted.loc[:, list(SAMPLE_COLUMNS)].to_string(index=False))
    else:
        print(accepted.loc[:, list(SAMPLE_COLUMNS)].head(MAX_SAMPLE_ROWS).to_string(index=False))
        print(f"[TRUNCATED] accepted rows limited to {MAX_SAMPLE_ROWS}")

    counts = failure_counts(monthly)
    print(
        "\n[FAILURE_MODE_COUNTS] "
        f"null_raw_close_shape={counts.null_raw_close_shape} "
        f"acquisition_failure={counts.acquisition_failure} "
        f"other_required_field_failure={counts.other_required_field_failure} "
        f"post_acquisition_processing_failure={counts.post_acquisition_processing_failure}"
    )
    return counts


def _date_distribution(frame: pd.DataFrame) -> pd.Series | None:
    if isinstance(frame.index, pd.DatetimeIndex):
        values = pd.Series(frame.index, index=frame.index)
    else:
        column = next((name for name in ("date", "timestamp", "datetime") if name in frame), None)
        if column is None:
            return None
        values = pd.to_datetime(frame[column], errors="coerce")
    return values.dt.date.value_counts(dropna=False).sort_index()


def _compare_monthly_snapshots(data_root: Path) -> None:
    affected_path = artifact_path(FAMILY_RUN_ID, "monthly", data_root)
    previous_path = artifact_path(PREVIOUS_FAMILY_RUN_ID, "monthly", data_root)
    affected = storage.load_parquet(affected_path)
    previous = storage.load_parquet(previous_path)
    for label, path, frame in (
        ("affected", affected_path, affected),
        ("previous", previous_path, previous),
    ):
        if "symbol" not in frame:
            raise DiagnosticFailure(f"{label} monthly snapshot missing symbol column: {path}")
        symbols = set(frame["symbol"].dropna().astype(str).str.strip().str.upper())
        print(f"[SNAPSHOT] {label} path={path} rows={len(frame)} unique_symbols={len(symbols)}")
        distribution = _date_distribution(frame)
        if distribution is None:
            print(f"[SNAPSHOT_DATES] {label} unavailable")
        else:
            print(f"[SNAPSHOT_DATES] {label}\n{distribution.to_string()}")

    affected_symbols = set(affected["symbol"].dropna().astype(str).str.strip().str.upper())
    previous_symbols = set(previous["symbol"].dropna().astype(str).str.strip().str.upper())
    print(
        "[SNAPSHOT_COMPARISON] "
        f"intersection={len(affected_symbols & previous_symbols)} "
        f"removed={len(previous_symbols - affected_symbols)} "
        f"added={len(affected_symbols - previous_symbols)}"
    )
    print(f"[AFFECTED_MONTHLY_SYMBOLS] {sorted(affected_symbols)}")


def run_diagnostic(*, data_root: Path = DATA) -> str:
    """Read and report persisted incident evidence without invoking write APIs."""
    manifest_path = (
        data_root / "_health" / "stocks_eod_family" / FAMILY_RUN_ID / "manifest.parquet"
    )
    try:
        manifest = storage.load_parquet(manifest_path)
    except Exception as exc:
        raise DiagnosticFailure(f"unable to read health manifest {manifest_path}: {exc}") from exc
    monthly = filter_monthly_intervals(manifest)
    counts = _print_manifest_analysis(monthly)

    try:
        _compare_monthly_snapshots(data_root)
    except Exception as exc:
        print(f"[SNAPSHOT_COMPARISON_UNAVAILABLE] {type(exc).__name__}: {exc}")

    classification = classify_failure(counts)
    print(f"[CLASSIFICATION] {classification}")
    return classification


def main() -> int:
    try:
        run_diagnostic()
    except DiagnosticFailure as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
