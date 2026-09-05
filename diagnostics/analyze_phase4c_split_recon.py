#!/usr/bin/env python3
"""Performance-blind Phase 4C chronological split reconnaissance.

This is a design utility, not policy calibration.  It reads only a narrow
projection of the Phase 4B episode population and reports observability and
sample-size properties of natural calendar cutoffs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from diagnostics.analyze_stock_options_phase4b import (
    CANDIDATE_GRIDS,
    candidate_policies,
)

SUPPORTED = {
    "stocks_c_dwm_all": ("DWM_1x", "DWM_2x", "DWM_3x"),
    "stocks_b_wmq_all": ("WMQ_1x", "WMQ_2x"),
}
HORIZON_DAYS = {"DWM_1x": 30, "DWM_2x": 60, "DWM_3x": 90, "WMQ_1x": 90, "WMQ_2x": 180}
FORBIDDEN_COLUMNS = (
    "directional_return",
    "mfe",
    "mae",
    "hit_rate",
    "mean_return",
    "median_return",
    "return_quantiles",
    "bootstrap_interval",
)
PROJECTED_COLUMNS = (
    "combo",
    "symbol",
    "direction",
    "episode_id",
    "episode_start_date",
    "effective_entry_date",
    "episode_end_date",
    "episode_observation_count",
    "horizon_id",
    "outcome_price_asof",
    "theoretically_mature",
    "terminal_covered",
    "coverage_status",
    "historical_participation_pass",
    "lower_sigvol_tier",
    "middle_sigvol_tier",
    "lower_ratio",
    "middle_ratio",
    "upper_has_wyckoff",
    "corporate_action_flag",
)
STALE = "SYMBOL_HISTORY_STALE_BEFORE_TARGET"
COHORTS = (
    "WHOLE_POLICY_COHORT",
    "CURRENT_PASS_RETAINED",
    "MARGINAL_NEW_ADMISSIONS",
    "STILL_BLOCKED",
)
OUTPUTS = (
    "phase4c_split_recon_summary.json",
    "phase4c_split_recon_report.md",
    "candidate_cutoff_counts.csv",
    "candidate_cutoff_feasibility.csv",
    "candidate_cutoff_by_participation.csv",
    "candidate_cutoff_candidate_policy_counts.csv",
    "candidate_cutoff_temporal_span.csv",
)


def read_episode_population(path: Path) -> pd.DataFrame:
    """Read an explicit allowlist; forbidden values never enter memory."""
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(PROJECTED_COLUMNS) - names
    if missing:
        raise ValueError(
            f"Phase 4B episode population missing columns: {sorted(missing)}"
        )
    if set(PROJECTED_COLUMNS) & set(FORBIDDEN_COLUMNS):
        raise AssertionError("performance field requested by split reconnaissance")
    return pd.read_parquet(path, columns=list(PROJECTED_COLUMNS))


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    requested = set(frame.columns)
    contamination = requested & set(FORBIDDEN_COLUMNS)
    # In-memory callers (not the parquet path) may supply performance columns to
    # prove invariance; immediately project them away before any analysis.
    allowed = [c for c in PROJECTED_COLUMNS if c in frame]
    required = set(PROJECTED_COLUMNS) - {"outcome_price_asof"}
    if missing := required - set(allowed):
        raise ValueError(f"episode inputs missing: {sorted(missing)}")
    result = frame.loc[:, allowed].copy()
    for col in (
        "effective_entry_date",
        "episode_start_date",
        "episode_end_date",
        "outcome_price_asof",
    ):
        if col in result:
            result[col] = pd.to_datetime(result[col]).dt.normalize()
    result.attrs["discarded_forbidden"] = sorted(contamination)
    result = result[result.combo.isin(SUPPORTED)].copy()
    result = result[result.apply(lambda r: r.horizon_id in SUPPORTED[r.combo], axis=1)]
    return result


def candidate_cutoffs(frame: pd.DataFrame, combo: str) -> list[pd.Timestamp]:
    dates = pd.to_datetime(frame.loc[frame.combo.eq(combo), "effective_entry_date"])
    if dates.empty:
        return []
    # Actual represented months, expressed as auditable calendar month ends.
    return sorted(dates.dt.to_period("M").dt.to_timestamp("M").unique())


def _side(frame: pd.DataFrame, cutoff: pd.Timestamp, side: str) -> pd.DataFrame:
    entry = pd.to_datetime(frame.effective_entry_date)
    return frame[entry.le(cutoff) if side == "CALIBRATION" else entry.gt(cutoff)]


def _counts(part: pd.DataFrame) -> dict:
    mature = part.theoretically_mature.fillna(False).astype(bool)
    usable = part.terminal_covered.fillna(False).astype(bool)
    stale = part.coverage_status.eq(STALE)
    return {
        "total_episode_anchors": len(part),
        "theoretically_mature_episodes": int(mature.sum()),
        "usable_outcome_episodes": int(usable.sum()),
        "immature_episodes": int((~mature).sum()),
        "stale_history_censored": int(stale.sum()),
        "other_censored": int((mature & ~usable & ~stale).sum()),
        "corporate_action_flagged_usable": int(
            (usable & part.corporate_action_flag.fillna(False)).sum()
        ),
        "unique_symbols": part.symbol.nunique(),
        "unique_entry_dates": part.effective_entry_date.nunique(),
        "first_entry_date": part.effective_entry_date.min(),
        "last_entry_date": part.effective_entry_date.max(),
        "overall_usable_coverage": (
            int(usable.sum()) / int(mature.sum()) if mature.any() else np.nan
        ),
        "stale_history_rate": (
            int(stale.sum()) / int(mature.sum()) if mature.any() else np.nan
        ),
    }


def count_matrices(frame: pd.DataFrame, cutoffs: dict[str, list[pd.Timestamp]]):
    counts, participation, temporal = [], [], []
    for combo, horizons in SUPPORTED.items():
        for cutoff in cutoffs.get(combo, []):
            for direction in ("LONG", "SHORT"):
                for horizon in horizons:
                    base = frame[
                        (frame.combo == combo)
                        & (frame.direction == direction)
                        & (frame.horizon_id == horizon)
                    ]
                    for side in ("CALIBRATION", "HOLDOUT"):
                        part = _side(base, cutoff, side)
                        row = {
                            "candidate_cutoff": cutoff,
                            "combo": combo,
                            "direction": direction,
                            "horizon": horizon,
                            "split_side": side,
                        } | _counts(part)
                        counts.append(row)
                        mature = part.theoretically_mature.fillna(False).astype(bool)
                        usable = part.terminal_covered.fillna(False).astype(bool)
                        for status, mask in (
                            (
                                "CURRENT_PASS",
                                part.historical_participation_pass.astype(bool),
                            ),
                            (
                                "CURRENT_BLOCK",
                                ~part.historical_participation_pass.astype(bool),
                            ),
                        ):
                            denominator = int((mature & mask).sum())
                            u = int((usable & mask).sum())
                            participation.append(
                                {
                                    "candidate_cutoff": cutoff,
                                    "combo": combo,
                                    "direction": direction,
                                    "horizon": horizon,
                                    "split_side": side,
                                    "historical_cohort": status,
                                    "total_episode_anchors": int(mask.sum()),
                                    "theoretically_mature_episodes": denominator,
                                    "usable_outcome_episodes": u,
                                    "usable_coverage": (
                                        u / denominator if denominator else np.nan
                                    ),
                                }
                            )
                        if side == "HOLDOUT":
                            m = part[mature]
                            first, last = (
                                m.effective_entry_date.min(),
                                m.effective_entry_date.max(),
                            )
                            temporal.append(
                                {
                                    "candidate_cutoff": cutoff,
                                    "combo": combo,
                                    "direction": direction,
                                    "horizon": horizon,
                                    "holdout_first_entry_date": first,
                                    "holdout_last_mature_entry_date": last,
                                    "holdout_calendar_span_days": (
                                        (last - first).days if len(m) else 0
                                    ),
                                    "holdout_unique_entry_months": m.effective_entry_date.dt.to_period(
                                        "M"
                                    ).nunique(),
                                    "holdout_unique_weekly_evaluation_dates": (
                                        m.effective_entry_date.nunique()
                                        if combo == "stocks_b_wmq_all"
                                        else np.nan
                                    ),
                                    "short_temporal_span": (
                                        (m.effective_entry_date.nunique() < 8)
                                        if combo == "stocks_b_wmq_all"
                                        else (
                                            m.effective_entry_date.dt.to_period(
                                                "M"
                                            ).nunique()
                                            < 2
                                        )
                                    ),
                                }
                            )
    participation = pd.DataFrame(participation)
    coverage_keys = [
        "candidate_cutoff",
        "combo",
        "direction",
        "horizon",
        "split_side",
    ]
    coverage = participation.pivot(
        index=coverage_keys,
        columns="historical_cohort",
        values="usable_coverage",
    )
    difference = (
        (coverage.CURRENT_PASS - coverage.CURRENT_BLOCK)
        .abs()
        .rename("pass_block_coverage_difference")
    )
    participation = participation.merge(difference.reset_index(), on=coverage_keys)
    participation["coverage_imbalance"] = participation[
        "pass_block_coverage_difference"
    ].gt(0.05)
    return pd.DataFrame(counts), participation, pd.DataFrame(temporal)


def policy_counts(
    frame: pd.DataFrame, cutoffs: dict[str, list[pd.Timestamp]]
) -> pd.DataFrame:
    candidates = candidate_policies(frame)
    rows = []
    for combo, horizons in SUPPORTED.items():
        for cutoff in cutoffs.get(combo, []):
            for direction in ("LONG", "SHORT"):
                for horizon in horizons:
                    base = candidates[
                        (candidates.combo == combo)
                        & (candidates.direction == direction)
                        & (candidates.horizon_id == horizon)
                    ]
                    for side in ("CALIBRATION", "HOLDOUT"):
                        part = _side(base, cutoff, side)
                        for (family, threshold), group in part.groupby(
                            ["threshold_family", "candidate_threshold"]
                        ):
                            for cohort in COHORTS:
                                mask = (
                                    group.candidate_pass.astype(bool)
                                    if cohort == "WHOLE_POLICY_COHORT"
                                    else group.policy_cohort.eq(cohort)
                                )
                                selected = group[mask]
                                usable = int(
                                    selected.terminal_covered.fillna(False).sum()
                                )
                                tier = (
                                    marginal_tier(usable, combo)
                                    if cohort == "MARGINAL_NEW_ADMISSIONS"
                                    else "NOT_APPLICABLE"
                                )
                                rows.append(
                                    {
                                        "candidate_cutoff": cutoff,
                                        "combo": combo,
                                        "direction": direction,
                                        "horizon": horizon,
                                        "threshold_family": family,
                                        "candidate_threshold": threshold,
                                        "split_side": side,
                                        "policy_cohort": cohort,
                                        "episode_count": len(selected),
                                        "usable_outcome_count": usable,
                                        "marginal_count_tier": tier,
                                        "wmq_lower_volume_tier": (
                                            marginal_tier(usable, combo, alternate=True)
                                            if cohort == "MARGINAL_NEW_ADMISSIONS"
                                            and combo == "stocks_b_wmq_all"
                                            else "NOT_APPLICABLE"
                                        ),
                                    }
                                )
    return pd.DataFrame(rows)


def marginal_tier(n: int, combo: str, alternate: bool = False) -> str:
    bounds = (
        (50, 30, 15) if alternate and combo == "stocks_b_wmq_all" else (100, 50, 25)
    )
    return (
        "ZERO"
        if n == 0
        else (
            "ROBUST_COUNT"
            if n >= bounds[0]
            else (
                "MODERATE_COUNT"
                if n >= bounds[1]
                else "THIN_COUNT" if n >= bounds[2] else "VERY_THIN"
            )
        )
    )


def feasibility(
    counts: pd.DataFrame, participation: pd.DataFrame, temporal: pd.DataFrame
) -> pd.DataFrame:
    merged = counts.merge(
        participation.pivot(
            index=["candidate_cutoff", "combo", "direction", "horizon", "split_side"],
            columns="historical_cohort",
            values="usable_outcome_episodes",
        )
        .add_prefix("usable_")
        .reset_index(),
        how="left",
    )
    keys = ["candidate_cutoff", "combo", "direction", "horizon"]
    wide = merged.pivot(
        index=keys, columns="split_side", values="usable_outcome_episodes"
    ).reset_index()
    hold = merged[merged.split_side.eq("HOLDOUT")][
        keys
        + [
            "usable_CURRENT_PASS",
            "usable_CURRENT_BLOCK",
            "theoretically_mature_episodes",
            "stale_history_censored",
        ]
    ]
    result = wide.merge(hold, on=keys).merge(
        temporal[keys + ["short_temporal_span"]], on=keys
    )
    for tier in ("preferred", "minimum"):
        vals = {
            "stocks_c_dwm_all": (500, 100) if tier == "preferred" else (250, 50),
            "stocks_b_wmq_all": (150, 30) if tier == "preferred" else (75, 20),
        }
        result[f"{tier}_feasible"] = result.apply(
            lambda r: r.CALIBRATION >= vals[r.combo][0]
            and r.HOLDOUT >= vals[r.combo][0]
            and r.usable_CURRENT_PASS >= vals[r.combo][1]
            and r.usable_CURRENT_BLOCK >= vals[r.combo][1],
            axis=1,
        )
    # Coverage warnings remain observability-only.
    cov = participation.pivot(
        index=["candidate_cutoff", "combo", "direction", "horizon", "split_side"],
        columns="historical_cohort",
        values="usable_coverage",
    ).reset_index()
    cov["coverage_imbalance"] = (cov.CURRENT_PASS - cov.CURRENT_BLOCK).abs().gt(0.05)
    result = result.merge(
        cov[cov.split_side.eq("HOLDOUT")].drop(columns="split_side"), on=keys
    )
    for tier in ("preferred", "minimum"):
        result[f"all_horizons_{tier}"] = result.groupby(
            ["candidate_cutoff", "combo", "direction"]
        )[f"{tier}_feasible"].transform("all")
    result["limiting_count"] = result[
        ["CALIBRATION", "HOLDOUT", "usable_CURRENT_PASS", "usable_CURRENT_BLOCK"]
    ].min(axis=1)
    result["limiting_reason"] = result[
        ["CALIBRATION", "HOLDOUT", "usable_CURRENT_PASS", "usable_CURRENT_BLOCK"]
    ].idxmin(axis=1)
    result["limiting_horizon"] = result.loc[
        result.groupby(["candidate_cutoff", "combo", "direction"])["limiting_count"]
        .transform("min")
        .eq(result.limiting_count),
        "horizon",
    ]
    return result


def recommendations(feasible: pd.DataFrame) -> list[dict]:
    rows = []
    for (combo, direction), group in feasible.groupby(["combo", "direction"]):
        item = {"combo": combo, "direction": direction}
        per_cutoff = group.groupby("candidate_cutoff").agg(
            preferred=("all_horizons_preferred", "all"),
            minimum=("all_horizons_minimum", "all"),
        )
        for tier in ("preferred", "minimum"):
            dates = per_cutoff.index[per_cutoff[tier]]
            item[f"earliest_{tier}_cutoff"] = (
                dates.min().date().isoformat() if len(dates) else None
            )
            item[f"latest_{tier}_cutoff"] = (
                dates.max().date().isoformat() if len(dates) else None
            )
            item[f"{tier}_status"] = "FEASIBLE" if len(dates) else "NO_FEASIBLE_SPLIT"
        rows.append(item)
    return rows


def common_direction_cutoffs(feasible: pd.DataFrame) -> list[dict]:
    """Describe, without forcing, each combo's LONG/SHORT feasible intersection."""
    rows = []
    for combo, group in feasible.groupby("combo"):
        item = {"combo": combo}
        by = group.groupby(["candidate_cutoff", "direction"]).agg(
            preferred=("all_horizons_preferred", "all"),
            minimum=("all_horizons_minimum", "all"),
        )
        for tier in ("preferred", "minimum"):
            wide = by[tier].unstack("direction", fill_value=False)
            common = wide.index[
                wide.reindex(columns=["LONG", "SHORT"], fill_value=False).all(axis=1)
            ]
            item[f"latest_common_{tier}_cutoff"] = (
                common.max().date().isoformat() if len(common) else None
            )
            item[f"common_{tier}_status"] = (
                "FEASIBLE" if len(common) else "NO_FEASIBLE_SPLIT"
            )
        rows.append(item)
    return rows


def run(input_dir: Path, output: Path) -> dict:
    source = input_dir / "phase4b_episode_population.parquet"
    frame = prepare(read_episode_population(source))
    asofs = frame.outcome_price_asof.dropna().unique()
    if len(asofs) != 1:
        raise ValueError("authoritative dataset_asof must be singular")
    cutoffs = {combo: candidate_cutoffs(frame, combo) for combo in SUPPORTED}
    counts, participation, temporal = count_matrices(frame, cutoffs)
    policies = policy_counts(frame, cutoffs)
    feasible = feasibility(counts, participation, temporal)
    recs = recommendations(feasible)
    summary = {
        "phase": "phase4c_split_recon",
        "performance_blind_recon": True,
        "dataset_asof": pd.Timestamp(asofs[0]).date().isoformat(),
        "forbidden_columns_checked": list(FORBIDDEN_COLUMNS),
        "projected_columns": list(PROJECTED_COLUMNS),
        "candidate_grids": CANDIDATE_GRIDS,
        "supported_horizons": SUPPORTED,
        "excluded": {
            "stocks_b_wmq_all": ["WMQ_3x"],
            "stocks_a_mqy_all": "NOT YET MATURE / NOT ELIGIBLE FOR PHASE 4C",
        },
        "recommendations": recs,
        "common_long_short_cutoffs": common_direction_cutoffs(feasible),
        "outputs": OUTPUTS,
    }
    output.mkdir(parents=True, exist_ok=True)
    for data, name in (
        (counts, OUTPUTS[2]),
        (feasible, OUTPUTS[3]),
        (participation, OUTPUTS[4]),
        (policies, OUTPUTS[5]),
        (temporal, OUTPUTS[6]),
    ):
        data.to_csv(output / name, index=False)
    (output / OUTPUTS[0]).write_text(json.dumps(summary, indent=2, default=list) + "\n")
    _report(output / OUTPUTS[1], summary, frame, feasible, policies)
    return summary


def _report(
    path: Path,
    summary: dict,
    frame: pd.DataFrame,
    feasible: pd.DataFrame,
    policies: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 4C split reconnaissance",
        "",
        "> Performance-blind design reconnaissance only; no outcome values or threshold performance were read.",
        "",
        f"**Authoritative dataset_asof:** {summary['dataset_asof']}",
        "",
        "## Entry ranges",
    ]
    ranges = frame.groupby(["combo", "direction"]).effective_entry_date.agg(
        ["min", "max"]
    )
    lines += ["", ranges.to_markdown(), "", "## Chronology/sample-size cutoffs", ""]
    for rec in summary["recommendations"]:
        lines.append(
            f"- **{rec['combo']} {rec['direction']}** — latest preferred: {rec['latest_preferred_cutoff'] or 'NO_FEASIBLE_SPLIT'}; latest minimum: {rec['latest_minimum_cutoff'] or 'NO_FEASIBLE_SPLIT'}. Design space: preferred {rec['earliest_preferred_cutoff']}–{rec['latest_preferred_cutoff']}; minimum {rec['earliest_minimum_cutoff']}–{rec['latest_minimum_cutoff']}."
        )
    lines += ["", "## Common LONG/SHORT view", ""]
    for item in summary["common_long_short_cutoffs"]:
        lines.append(
            f"- **{item['combo']}** — latest common preferred: "
            f"{item['latest_common_preferred_cutoff'] or 'NO_FEASIBLE_SPLIT'}; "
            f"latest common minimum: {item['latest_common_minimum_cutoff'] or 'NO_FEASIBLE_SPLIT'}."
        )
    lines += [
        "",
        "The all-horizon decision uses 30/60/90d together for D/W/M and 90/180d together for W/M/Q. The `limiting_horizon`, count, temporal-span, and coverage-warning fields are preserved in the CSVs. A common LONG/SHORT cutoff is reasonable only where the intersection of their feasible cutoff sets is non-empty; humans should precommit within that intersection, preferring its latest date if recency is the objective.",
        "",
        "M/Q/Y: **NOT YET MATURE / NOT ELIGIBLE FOR PHASE 4C**. W/M/Q 270d is excluded.",
        "",
        "## Candidate grids",
        "",
        "- MODERATE: 0.025, 0.05, 0.075, 0.10, 0.15, **0.20**, 0.25",
        "- STRONG + upper available: 0.02, 0.025, 0.03, 0.04, 0.05",
        "- STRONG + upper unavailable: 0.025, 0.05, 0.075, 0.10",
        "",
        "Every candidate/cohort count, including zero/thin marginal admissions, is in `candidate_cutoff_candidate_policy_counts.csv`. No return, MFE, MAE, hit-rate, quantile, effect, or bootstrap field is projected.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("diagnostic_artifacts/phase4b")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostic_artifacts/phase4c_split_recon"),
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
