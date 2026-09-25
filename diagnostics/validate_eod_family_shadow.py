"""Read-only structural validator for a PR-A EOD family shadow run."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from core import storage
from etl.eod_family import FAMILY_INTERVALS, shadow_manifest_path


TERMINAL_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")


def _equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0))
    except (TypeError, ValueError):
        return left == right


def _repair_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise AssertionError("repair_provenance_json must contain a JSON object")
    return parsed


def validate_manifest(manifest: pd.DataFrame, symbols: list[str]) -> dict[str, Any]:
    """Fail closed on structural or reconciliation-contract violations."""
    expected_symbols = [symbol.strip().upper() for symbol in symbols]
    intervals = manifest.loc[manifest["record_type"].eq("interval")].copy()
    summaries = manifest.loc[manifest["record_type"].eq("run_summary")]
    assert len(summaries) == 1, "manifest must contain exactly one run_summary row"
    assert set(intervals["symbol"]) == set(expected_symbols), "manifest symbol set mismatch"

    for symbol in expected_symbols:
        rows = intervals.loc[intervals["symbol"].eq(symbol)]
        assert len(rows) == 3, f"{symbol}: expected exactly three interval rows"
        assert set(rows["interval"]) == set(FAMILY_INTERVALS), (
            f"{symbol}: intervals must be exactly {FAMILY_INTERVALS}"
        )
        assert rows["requests_attempted"].eq(3).all(), (
            f"{symbol}: requests_attempted must equal three"
        )
        successful_intervals = int(rows["acquisition_status"].eq("success").sum())
        failed_intervals = 3 - successful_intervals
        assert rows["requests_succeeded"].eq(successful_intervals).all(), (
            f"{symbol}: successful request count disagrees with interval states"
        )
        assert rows["requests_failed"].eq(failed_intervals).all(), (
            f"{symbol}: failed request count disagrees with interval states"
        )
        assert rows["reconciliation_provider_requests"].eq(0).all(), (
            f"{symbol}: reconciliation made a provider request"
        )
        assert rows["acquisition_status"].isin(["success", "failed"]).all(), (
            f"{symbol}: unknown or missing acquisition state"
        )
        for _, row in rows.iterrows():
            raw_values = [row[f"raw_{field}"] for field in TERMINAL_FIELDS]
            if row["acquisition_status"] == "failed":
                assert pd.notna(row["acquisition_error"]), (
                    f"{symbol}/{row['interval']}: failed interval lacks an error"
                )
                assert all(pd.isna(value) for value in raw_values), (
                    f"{symbol}/{row['interval']}: non-empty observation mislabeled failed"
                )
            else:
                assert pd.isna(row["acquisition_error"]), (
                    f"{symbol}/{row['interval']}: successful acquisition has an error"
                )
                assert row["quality_status"] in {"valid", "invalid"}, (
                    f"{symbol}/{row['interval']}: acquired frame lacks quality result"
                )

            provenance = _repair_provenance(row["repair_provenance_json"])
            changed = [field for field in TERMINAL_FIELDS
                       if not _equal(row[f"raw_{field}"], row[f"finalized_{field}"])]
            if row["interval"] in {"1wk", "1mo"}:
                assert not changed, f"{symbol}/{row['interval']}: W/M target was modified"
                assert row["reconciliation_status"] == "not_needed", (
                    f"{symbol}/{row['interval']}: unsupported reconciliation target"
                )
                assert provenance.get("status") == "native_unmodified", (
                    f"{symbol}/{row['interval']}: W/M provenance is not native-unmodified"
                )
            elif changed:
                assert changed == ["close"], f"{symbol}/1d: repair changed {changed}"
                assert row["acquisition_status"] == "success" and row["quality_status"] == "invalid", (
                    f"{symbol}/1d: repaired target was not an acquired malformed observation"
                )
                assert pd.isna(row["raw_close"]), f"{symbol}/1d: repair overwrote an observed Close"
                assert provenance.get("data_quality_status") == "repaired", (
                    f"{symbol}/1d: changed Close lacks repaired provenance"
                )
                assert row["reconciliation_status"] == "canonical", (
                    f"{symbol}/1d: repair lacks canonical reconciliation"
                )

        daily = rows.loc[rows["interval"].eq("1d")].iloc[0]
        if daily["reconciliation_status"] == "canonical":
            sources = str(daily["canonical_source_intervals"]).split("+")
            assert set(sources).issubset({"1wk", "1mo"}), (
                f"{symbol}: canonical decision contains unsupported source"
            )
            for source in sources:
                evidence = rows.loc[rows["interval"].eq(source)].iloc[0]
                assert evidence["terminal_session"] == daily["terminal_session"], (
                    f"{symbol}: canonical {source} session contradicts daily target"
                )

    unique_families = intervals.drop_duplicates(["family_run_id", "symbol"])
    summary = summaries.iloc[0]
    expected_attempts = 3 * len(expected_symbols)
    assert int(unique_families["requests_attempted"].sum()) == expected_attempts
    assert int(summary["requests_attempted"]) == expected_attempts
    assert int(summary["reconciliation_provider_requests"]) == 0
    assert int(summary["requests_succeeded"]) + int(summary["requests_failed"]) == expected_attempts

    needs_indicators = False
    for _, row in intervals.iterrows():
        provenance = _repair_provenance(row["repair_provenance_json"])
        if row["quality_status"] == "valid" or provenance.get("data_quality_status") == "repaired":
            needs_indicators = True
            break
    return {
        "intervals": intervals,
        "run_summary": summary,
        "needs_indicators": needs_indicators,
    }


def _markdown(result: dict[str, Any], run_id: str, symbols: list[str], indicator: pd.DataFrame | None) -> str:
    intervals = result["intervals"]
    summary = result["run_summary"]
    display_columns = [
        "symbol", "interval", "acquisition_status", "quality_status", "quality_reason",
        "raw_close", "close_is_null", "terminal_session", "reconciliation_status",
        "reconciliation_tier", "canonical_source_intervals", "canonical_observed_close",
        "finalized_close", "validity_pattern", "close_null_pattern", "fetch_started_at",
        "fetch_completed_at", "family_duration_seconds",
    ]
    families = intervals.drop_duplicates(["family_run_id", "symbol"])
    daily = intervals.loc[intervals["interval"].eq("1d")]
    lines = [
        "# EOD family shadow validation", "", f"- **RUN_ID:** `{run_id}`",
        f"- **Requested symbols:** `{', '.join(symbols)}`",
        f"- **Provider attempts/successes/failures:** {int(summary['requests_attempted'])} / "
        f"{int(summary['requests_succeeded'])} / {int(summary['requests_failed'])}",
        f"- **Reconciliation provider requests:** {int(summary['reconciliation_provider_requests'])}",
        f"- **First fetch start:** `{summary['fetch_started_at']}`",
        f"- **Last fetch completion:** `{summary['fetch_completed_at']}`",
        f"- **Acquisition wall seconds:** {summary['family_duration_seconds']}", "",
        "## Per symbol / interval", "", "```text",
        intervals[display_columns].to_string(index=False), "```", "",
        "## Validity-pattern counts", "", "```text",
        families["validity_pattern"].value_counts(dropna=False).to_string(), "```", "",
        "## Close-null-pattern counts", "", "```text",
        families["close_null_pattern"].value_counts(dropna=False).to_string(), "```", "",
        "## Reconciliation status / tier counts", "", "```text",
        daily.groupby(["reconciliation_status", "reconciliation_tier"], dropna=False).size().to_string(),
        "```", "", "## Indicator rows by symbol / interval", "", "```text",
        (indicator.groupby(["symbol", "interval"]).size().to_string()
         if indicator is not None and not indicator.empty else "none"), "```", "",
    ]
    return "\n".join(lines)


def validate_run(run_id: str, symbols: list[str], output_dir: Path) -> None:
    manifest_path = shadow_manifest_path(run_id)
    assert storage.exists(manifest_path), f"manifest does not exist: {manifest_path}"
    manifest = storage.load_parquet(manifest_path)
    result = validate_manifest(manifest, symbols)

    indicator_path = manifest_path.with_name("indicator_terminals.parquet")
    indicator = storage.load_parquet(indicator_path) if storage.exists(indicator_path) else None
    if result["needs_indicators"]:
        assert indicator is not None and not indicator.empty, (
            "indicator artifact required for at least one valid/repaired interval"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_dir / "manifest.parquet", index=False)
    if indicator is not None:
        indicator.to_parquet(output_dir / "indicator_terminals.parquet", index=False)
    result["intervals"].to_csv(output_dir / "interval_summary.csv", index=False)
    summary = _markdown(result, run_id, symbols, indicator)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    github_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_run(args.run_id, args.symbols, args.output_dir)


if __name__ == "__main__":
    main()
