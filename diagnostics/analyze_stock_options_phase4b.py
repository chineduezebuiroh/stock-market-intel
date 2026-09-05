#!/usr/bin/env python3
"""Phase 4B participation/outcome characterization (local outputs, read only).

This stage consumes validated Phase 3B populations and the Phase 4A outcome
artifact.  It deliberately has no storage or market-data dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.analyze_stock_options_participation import (
    COMBO_SPECS,
    build_directional_opportunities,
    scenario_pass,
)
from diagnostics.analyze_stock_options_phase3b import validate_population
from diagnostics.outcomes.specs import CoverageStatus, PolicyMode

SUPPORTED_HORIZONS = {
    "stocks_c_dwm_all": frozenset({"DWM_1x", "DWM_2x", "DWM_3x"}),
    "stocks_b_wmq_all": frozenset({"WMQ_1x", "WMQ_2x"}),
    "stocks_a_mqy_all": frozenset(),
}
CADENCE = {"stocks_c_dwm_all": "DAILY", "stocks_b_wmq_all": "WEEKLY"}
CANDIDATE_GRIDS = {
    "MODERATE": (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25),
    "STRONG_UPPER_AVAILABLE": (0.02, 0.025, 0.03, 0.04, 0.05),
    "STRONG_UPPER_UNAVAILABLE": (0.025, 0.05, 0.075, 0.10),
}
CURRENT_THRESHOLDS = {
    "MODERATE": 0.25,
    "STRONG_UPPER_AVAILABLE": 0.05,
    "STRONG_UPPER_UNAVAILABLE": 0.10,
}
OUTPUT_FILES = (
    "phase4b_summary.json",
    "phase4b_report.md",
    "phase4b_episode_population.parquet",
    "phase4b_observation_population.parquet",
    "historical_pass_block_outcomes.csv",
    "historical_pass_block_effects.csv",
    "candidate_policy_outcomes.csv",
    "candidate_policy_marginal_cohorts.csv",
    "outcomes_by_sigvol_route.csv",
    "outcomes_by_entry_period.csv",
    "coverage_by_policy_cohort.csv",
    "corporate_action_robustness.csv",
    "bootstrap_uncertainty.csv",
    "episode_construction_audit.csv",
    "candidate_admission_counts.csv",
)


def construct_preparticipation_episodes(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify episodes using adjacency in each combo's canonical date sequence."""
    required = {"combo", "direction", "symbol", "lower_date", "pre_participation"}
    if missing := required - set(frame):
        raise AssertionError(f"episode inputs missing: {sorted(missing)}")
    if frame.duplicated(["combo", "direction", "symbol", "lower_date"]).any():
        raise AssertionError("duplicate canonical population keys")
    pieces = []
    for combo, whole in frame.groupby("combo", sort=False):
        dates = sorted(pd.to_datetime(whole.lower_date).dt.normalize().unique())
        position = {value: number for number, value in enumerate(dates)}
        pre = whole[whole.pre_participation.astype(bool)].copy()
        pre["lower_date"] = pd.to_datetime(pre.lower_date).dt.normalize()
        pre["_position"] = pre.lower_date.map(position)
        pre = pre.sort_values(["symbol", "direction", "_position"])
        start = pre.groupby(["symbol", "direction"])["_position"].diff().ne(1)
        pre["episode_number"] = (
            start.groupby([pre.symbol, pre.direction]).cumsum().astype(int)
        )
        pre["episode_id"] = (
            combo
            + ":"
            + pre.symbol.astype(str)
            + ":"
            + pre.direction
            + ":"
            + pre.episode_number.astype(str)
        )
        pieces.append(pre.drop(columns="_position"))
    observations = (
        pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()
    )
    observations["episode_start_date"] = observations.groupby(
        "episode_id"
    ).lower_date.transform("min")
    observations["episode_end_date"] = observations.groupby(
        "episode_id"
    ).lower_date.transform("max")
    observations["episode_observation_count"] = observations.groupby(
        "episode_id"
    ).symbol.transform("size")
    observations["episode_anchor"] = observations.lower_date.eq(
        observations.episode_start_date
    )
    anchors = observations[observations.episode_anchor].copy()
    if (
        anchors.episode_id.duplicated().any()
        or len(anchors) != observations.episode_id.nunique()
    ):
        raise AssertionError("invalid episode construction")
    return observations, anchors


def candidate_policies(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate only the three predeclared participation-threshold families."""
    rows = []
    for family, grid in CANDIDATE_GRIDS.items():
        for threshold in grid:
            kwargs = dict(strong_available=0.05, strong_unavailable=0.10, moderate=0.25)
            kwargs[
                {
                    "MODERATE": "moderate",
                    "STRONG_UPPER_AVAILABLE": "strong_available",
                    "STRONG_UPPER_UNAVAILABLE": "strong_unavailable",
                }[family]
            ] = threshold
            part = frame.copy()
            lower = _route_pass(part, "lower", **kwargs)
            middle = _route_pass(part, "middle", **kwargs)
            expected = scenario_pass(part, **kwargs)
            if not (lower | middle).equals(expected):
                raise AssertionError("counterfactual participation evaluation drift")
            part["policy_mode"] = PolicyMode.COUNTERFACTUAL_PARTICIPATION.value
            part["threshold_family"] = family
            part["candidate_threshold"] = threshold
            part["candidate_pass"] = expected
            part["candidate_admission_route"] = np.select(
                [lower & middle, lower, middle],
                ["BOTH", "LOWER_ONLY", "MIDDLE_ONLY"],
                default="NEITHER",
            )
            historical = part.historical_participation_pass.astype(bool)
            part["policy_cohort"] = np.select(
                [expected & historical, expected & ~historical, ~expected],
                ["CURRENT_PASS_RETAINED", "MARGINAL_NEW_ADMISSIONS", "STILL_BLOCKED"],
                default="INVALID",
            )
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _route_pass(
    frame: pd.DataFrame,
    route: str,
    *,
    moderate: float,
    strong_available: float,
    strong_unavailable: float,
) -> pd.Series:
    tier, ratio = frame[f"{route}_sigvol_tier"], frame[f"{route}_ratio"]
    threshold = np.select(
        [tier.eq(1), tier.eq(2) & frame.upper_has_wyckoff, tier.eq(2)],
        [moderate, strong_available, strong_unavailable],
        default=np.nan,
    )
    return tier.isin([1, 2]) & ratio.gt(threshold)


def summarize_outcomes(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mature = part[part.theoretically_mature.astype(bool)]
        usable = part[part.terminal_covered.astype(bool)]
        path = part[part.path_covered.astype(bool)]
        result = dict(zip(groups, keys)) | {
            "n_total": len(part),
            "n_mature": len(mature),
            "n_usable": len(usable),
            "coverage_rate": len(usable) / len(mature) if len(mature) else np.nan,
            "stale_history_count": int(
                part.coverage_status.eq(
                    CoverageStatus.SYMBOL_HISTORY_STALE_BEFORE_TARGET.value
                ).sum()
            ),
            "other_censoring_count": int(
                len(mature)
                - len(usable)
                - part.coverage_status.eq(
                    CoverageStatus.SYMBOL_HISTORY_STALE_BEFORE_TARGET.value
                ).sum()
            ),
            "corporate_action_flagged_count": int(part.corporate_action_flag.sum()),
        }
        for column in ("directional_return", "mfe", "mae"):
            values = (usable if column == "directional_return" else path)[
                column
            ].dropna()
            for stat, value in (
                ("mean", values.mean()),
                ("median", values.median()),
                ("p25", values.quantile(0.25)),
                ("p75", values.quantile(0.75)),
            ):
                result[f"{column}_{stat}"] = value
        returns = usable.directional_return.dropna()
        result["hit_rate"] = returns.gt(0).mean()
        result["return_ge_5pct_rate"] = (
            returns.ge(0.05).mean() if len(returns) >= 20 else np.nan
        )
        result["return_le_minus_5pct_rate"] = (
            returns.le(-0.05).mean() if len(returns) >= 20 else np.nan
        )
        rows.append(result)
    return pd.DataFrame(rows)


def effects(
    summary: pd.DataFrame, group_keys: list[str], cohort: str, left: str, right: str
) -> pd.DataFrame:
    metrics = [
        "directional_return_mean",
        "directional_return_median",
        "hit_rate",
        "mfe_median",
        "mae_median",
    ]
    wide = summary[summary[cohort].isin([left, right])].pivot_table(
        index=group_keys, columns=cohort, values=metrics
    )
    records = []
    for index, row in wide.iterrows():
        index = index if isinstance(index, tuple) else (index,)
        record = dict(zip(group_keys, index)) | {"comparison": f"{left}_MINUS_{right}"}
        for metric in metrics:
            record[f"delta_{metric}"] = row.get((metric, left), np.nan) - row.get(
                (metric, right), np.nan
            )
        records.append(record)
    return pd.DataFrame(records)


def symbol_cluster_bootstrap(
    frame: pd.DataFrame, groups: list[str], seed: int = 4204, replications: int = 1000
) -> pd.DataFrame:
    """Deterministic symbol-cluster bootstrap; dates remain a documented limitation."""
    rng = np.random.default_rng(seed)
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        usable = part[part.terminal_covered.astype(bool)].dropna(
            subset=["directional_return"]
        )
        symbols = usable.symbol.unique()
        draws = []
        if len(symbols):
            buckets = {
                s: usable[usable.symbol.eq(s)].directional_return.to_numpy()
                for s in symbols
            }
            for _ in range(replications):
                sample = np.concatenate(
                    [
                        buckets[s]
                        for s in rng.choice(symbols, len(symbols), replace=True)
                    ]
                )
                draws.append((sample.mean(), np.median(sample), (sample > 0).mean()))
        record = dict(zip(groups, keys if isinstance(keys, tuple) else (keys,))) | {
            "bootstrap_seed": seed,
            "bootstrap_replications": replications,
            "cluster": "symbol",
            "n_symbols": len(symbols),
        }
        for pos, metric in enumerate(
            ("mean_directional_return", "median_directional_return", "hit_rate")
        ):
            values = np.asarray(draws)[:, pos] if draws else np.array([])
            record[f"{metric}_ci_low"] = (
                np.quantile(values, 0.025) if len(values) else np.nan
            )
            record[f"{metric}_ci_high"] = (
                np.quantile(values, 0.975) if len(values) else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _load(validated_root: Path, phase4a: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase4a_summary = json.loads(
        (phase4a / "phase4a_coverage_summary.json").read_text()
    )
    if phase4a_summary.get("phase") != "phase4a_coverage" or not phase4a_summary.get(
        "read_only"
    ):
        raise AssertionError("Phase 4A contract drift")
    directional = []
    for combo in COMBO_SPECS:
        root = validated_root / combo
        canonical = pd.read_parquet(root / "supported_observations.parquet")
        validate_population(
            combo, canonical, json.loads((root / "coverage_summary.json").read_text())
        )
        directional.append(build_directional_opportunities(canonical, combo))
    all_observations = pd.concat(directional, ignore_index=True)
    all_observations["lower_date"] = pd.to_datetime(
        all_observations.lower_date
    ).dt.normalize()
    outcomes = pd.read_parquet(phase4a / "outcome_coverage_observations.parquet")
    outcomes["lower_bar_date"] = pd.to_datetime(outcomes.lower_bar_date).dt.normalize()
    allowed = outcomes.apply(
        lambda row: row.horizon_id in SUPPORTED_HORIZONS.get(row.combo, ()), axis=1
    )
    outcomes = outcomes[allowed].copy()
    keys = ["combo", "direction", "symbol", "lower_bar_date", "horizon_id"]
    if outcomes.duplicated(keys).any():
        raise AssertionError("duplicate Phase 4A outcome keys")
    return all_observations, outcomes


def run(validated_root: Path, phase4a: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    all_observations, outcomes = _load(validated_root, phase4a)
    observation_base, episode_anchors = construct_preparticipation_episodes(
        all_observations
    )
    feature_cols = [
        "combo",
        "direction",
        "symbol",
        "lower_date",
        "lower_sigvol_tier",
        "middle_sigvol_tier",
        "lower_ratio",
        "middle_ratio",
        "upper_has_wyckoff",
        "episode_id",
        "episode_start_date",
        "episode_end_date",
        "episode_observation_count",
        "episode_anchor",
    ]

    def merge(base: pd.DataFrame, unit: str) -> pd.DataFrame:
        result = outcomes.merge(
            base[feature_cols],
            left_on=["combo", "direction", "symbol", "lower_bar_date"],
            right_on=["combo", "direction", "symbol", "lower_date"],
            validate="many_to_one",
        )
        result["analysis_unit"] = unit
        return result

    observation_population = merge(observation_base, "OBSERVATION")
    episode_population = merge(episode_anchors, "EPISODE")
    observation_population.to_parquet(output / OUTPUT_FILES[3], index=False)
    episode_population.to_parquet(output / OUTPUT_FILES[2], index=False)
    observation_base[
        [
            "combo",
            "direction",
            "symbol",
            "lower_date",
            "episode_id",
            "episode_start_date",
            "episode_end_date",
            "episode_observation_count",
            "episode_anchor",
        ]
    ].to_csv(output / "episode_construction_audit.csv", index=False)

    both = pd.concat([episode_population, observation_population], ignore_index=True)
    both["historical_cohort"] = np.where(
        both.historical_participation_pass, "CURRENT_PASS", "CURRENT_BLOCK"
    )
    historical_groups = [
        "combo",
        "direction",
        "horizon_id",
        "analysis_unit",
        "historical_cohort",
    ]
    historical = summarize_outcomes(both, historical_groups)
    historical.to_csv(output / "historical_pass_block_outcomes.csv", index=False)
    effect = effects(
        historical,
        historical_groups[:-1],
        "historical_cohort",
        "CURRENT_PASS",
        "CURRENT_BLOCK",
    )
    effect.to_csv(output / "historical_pass_block_effects.csv", index=False)

    candidates = candidate_policies(both)
    candidate_groups = [
        "combo",
        "direction",
        "horizon_id",
        "analysis_unit",
        "threshold_family",
        "candidate_threshold",
        "policy_cohort",
    ]
    candidate_summary = summarize_outcomes(candidates, candidate_groups)
    # Whole-policy is intentionally an additional overlapping reporting cohort.
    whole = candidates[candidates.candidate_pass].copy()
    whole["policy_cohort"] = "WHOLE_POLICY_COHORT"
    candidate_summary = pd.concat(
        [candidate_summary, summarize_outcomes(whole, candidate_groups)],
        ignore_index=True,
    )
    candidate_summary.to_csv(output / "candidate_policy_outcomes.csv", index=False)
    marginal = candidate_summary[
        candidate_summary.policy_cohort.eq("MARGINAL_NEW_ADMISSIONS")
    ].copy()
    comparison_keys = candidate_groups[:-1]
    comparison_metrics = [
        "directional_return_mean",
        "directional_return_median",
        "hit_rate",
        "mfe_median",
        "mae_median",
    ]
    for comparison in ("CURRENT_PASS_RETAINED", "STILL_BLOCKED"):
        reference = candidate_summary[candidate_summary.policy_cohort.eq(comparison)][
            comparison_keys + comparison_metrics
        ].rename(
            columns={
                metric: f"{comparison.lower()}_{metric}"
                for metric in comparison_metrics
            }
        )
        marginal = marginal.merge(
            reference, on=comparison_keys, how="left", validate="one_to_one"
        )
        for metric in comparison_metrics:
            marginal[f"delta_{metric}_vs_{comparison.lower()}"] = (
                marginal[metric] - marginal[f"{comparison.lower()}_{metric}"]
            )
    marginal.to_csv(output / "candidate_policy_marginal_cohorts.csv", index=False)
    candidate_summary[
        [
            *candidate_groups,
            "n_total",
            "n_mature",
            "n_usable",
            "coverage_rate",
            "stale_history_count",
            "other_censoring_count",
            "corporate_action_flagged_count",
        ]
    ].to_csv(output / "coverage_by_policy_cohort.csv", index=False)
    candidates.groupby(candidate_groups, dropna=False).size().rename(
        "count"
    ).reset_index().to_csv(output / "candidate_admission_counts.csv", index=False)

    route_groups = [
        "combo",
        "direction",
        "horizon_id",
        "analysis_unit",
        "threshold_family",
        "candidate_threshold",
        "candidate_admission_route",
        "policy_cohort",
    ]
    summarize_outcomes(candidates, route_groups).to_csv(
        output / "outcomes_by_sigvol_route.csv", index=False
    )
    candidates["entry_period"] = (
        pd.to_datetime(candidates.effective_entry_date).dt.to_period("M").astype(str)
    )
    summarize_outcomes(candidates, candidate_groups + ["entry_period"]).to_csv(
        output / "outcomes_by_entry_period.csv", index=False
    )
    robust = []
    for excluded, data in ((False, both), (True, both[~both.corporate_action_flag])):
        x = summarize_outcomes(data, historical_groups)
        x["corporate_actions_excluded"] = excluded
        robust.append(x)
    pd.concat(robust).to_csv(output / "corporate_action_robustness.csv", index=False)
    boot_groups = ["combo", "direction", "horizon_id", "historical_cohort"]
    symbol_cluster_bootstrap(
        episode_population.assign(
            historical_cohort=np.where(
                episode_population.historical_participation_pass,
                "CURRENT_PASS",
                "CURRENT_BLOCK",
            )
        ),
        boot_groups,
    ).to_csv(output / "bootstrap_uncertainty.csv", index=False)

    summary = {
        "phase": "phase4b",
        "validation_scope": "cross_combo_phase3b",
        "read_only": True,
        "policy_mode": PolicyMode.COUNTERFACTUAL_PARTICIPATION.value,
        "entry_convention": "IMMUTABLE_SIGNAL_CLOSE",
        "candidate_grids": CANDIDATE_GRIDS,
        "supported_horizons": {k: sorted(v) for k, v in SUPPORTED_HORIZONS.items()},
        "cadence": CADENCE,
        "bootstrap": {"seed": 4204, "replications": 1000, "cluster": "symbol"},
        "production_threshold_changes": False,
        "outputs": OUTPUT_FILES,
    }
    (output / "phase4b_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(output, historical, effect, marginal)


def _write_report(
    output: Path, historical: pd.DataFrame, effect: pd.DataFrame, marginal: pd.DataFrame
) -> None:
    lines = [
        "# Phase 4B — participation outcome characterization",
        "",
        "> Descriptive/inferential evidence only. This report does not recommend or apply a production threshold.",
        "",
        "## Contracts and method",
        "",
        "Entry truth is immutable signal close; outcomes and censoring use the unchanged Phase 4A contract. Primary results use the first pre-participation-qualified observation in each episode; all canonical observations are secondary robustness. D/W/M uses daily canonical adjacency and W/M/Q weekly canonical adjacency. Bootstrap intervals resample symbols; common market-date dependence is a limitation.",
        "",
        "Supported horizons: D/W/M 30/60/90 calendar days; W/M/Q 90/180 days. W/M/Q 270 days and every M/Q/Y horizon are excluded as immature.",
        "",
        "## Historical PASS versus BLOCK",
        "",
        _markdown_table(historical),
        "",
        "## Effect sizes",
        "",
        _markdown_table(effect),
        "",
        "Positive directional return, MFE, and hit-rate deltas favor PASS. A less-negative (positive delta) MAE also favors PASS. Distribution quartiles and coverage must be considered; no scalar objective is used.",
        "",
        "## Pre-specified marginal admissions",
        "",
        _markdown_table(marginal),
        "",
        "Moderate: 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25. Strong+upper available: 0.02, 0.025, 0.03, 0.04, 0.05. Strong+upper unavailable: 0.025, 0.05, 0.075, 0.10.",
        "",
        "## Interpretation guardrails",
        "",
        "Review episode and observation rows separately, alongside censoring, corporate-action robustness, time-period tables, and symbol-cluster confidence intervals. Threshold families with adequate marginal sample size and stable coverage may proceed to Phase 4C chronological calibration/holdout; Phase 4B makes no policy selection.",
    ]
    (output / "phase4b_report.md").write_text("\n".join(lines) + "\n")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free Markdown table for the artifact report."""
    values = frame.replace({np.nan: ""}).astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    divider = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-root", type=Path, required=True)
    parser.add_argument("--phase4a-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.validated_root, args.phase4a_root, args.output_dir)
    print("[SAFETY] LOCAL ANALYSIS ONLY; NO S3 WRITES OR EXTERNAL PROVIDERS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
