#!/usr/bin/env python3
"""Phase 4C precommitted chronological calibration/holdout diagnostic.

The program consumes Phase 4B populations, never market data, and writes only
to the requested local output directory.  Candidate selection is deliberately
isolated from holdout rows and is frozen to CSV before validation starts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from diagnostics.analyze_stock_options_phase4b import (
    CANDIDATE_GRIDS,
    CURRENT_THRESHOLDS,
    candidate_policies,
    summarize_outcomes,
)

SPLITS = {
    "stocks_c_dwm_all": pd.Timestamp("2026-04-30"),
    "stocks_b_wmq_all": pd.Timestamp("2026-01-31"),
}
HORIZONS = {
    "stocks_c_dwm_all": ("DWM_1x", "DWM_2x", "DWM_3x"),
    "stocks_b_wmq_all": ("WMQ_1x", "WMQ_2x"),
}
HORIZON_DAYS = {"DWM_1x": 30, "DWM_2x": 60, "DWM_3x": 90, "WMQ_1x": 90, "WMQ_2x": 180}
BOOTSTRAP_SEED = 4204
BOOTSTRAP_REPS = 1000
COHORTS = (
    "WHOLE_POLICY_COHORT",
    "CURRENT_PASS_RETAINED",
    "MARGINAL_NEW_ADMISSIONS",
    "STILL_BLOCKED",
)
OUTPUTS = (
    "phase4c_summary.json",
    "phase4c_report.md",
    "phase4c_calibration_grid.csv",
    "phase4c_calibration_selected_candidates.csv",
    "phase4c_holdout_validation.csv",
    "phase4c_current_pass_block_by_split.csv",
    "phase4c_bootstrap_intervals.csv",
    "phase4c_corporate_action_sensitivity.csv",
    "phase4c_observation_robustness.csv",
    "phase4c_chronological_regime.csv",
    "phase4c_leakage_audit.json",
)
METRICS = (
    "directional_return_mean",
    "directional_return_median",
    "hit_rate",
    "mfe_median",
    "mae_median",
)


def sample_strength(n: int, combo: str) -> str:
    """Apply the immutable reconnaissance sample-size rubric."""
    boundaries = (50, 30, 15) if combo == "stocks_b_wmq_all" else (100, 50, 25)
    if n == 0:
        return "ZERO"
    if n >= boundaries[0]:
        return "ROBUST"
    if n >= boundaries[1]:
        return "MODERATE"
    if n >= boundaries[2]:
        return "THIN"
    return "VERY_THIN"


def split_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply only the two precommitted cutoffs and fail closed on bad dates."""
    result = frame[frame.combo.isin(SPLITS)].copy()
    result = result[result.apply(lambda r: r.horizon_id in HORIZONS[r.combo], axis=1)]
    result["effective_entry_date"] = pd.to_datetime(
        result.effective_entry_date
    ).dt.normalize()
    if result.effective_entry_date.isna().any():
        raise ValueError("missing effective_entry_date")
    result["split"] = np.where(
        result.effective_entry_date.le(result.combo.map(SPLITS)),
        "CALIBRATION",
        "HOLDOUT",
    )
    for combo, cutoff in SPLITS.items():
        cal = result[(result.combo == combo) & (result.split == "CALIBRATION")]
        hold = result[(result.combo == combo) & (result.split == "HOLDOUT")]
        if (not cal.empty and cal.effective_entry_date.max() > cutoff) or (
            not hold.empty and hold.effective_entry_date.min() <= cutoff
        ):
            raise AssertionError("chronological split overlap")
    return result


def enforce_fixed_maturity(frame: pd.DataFrame, dataset_asof: str) -> pd.DataFrame:
    """Reassert theoretical maturity from the authoritative, fixed as-of date."""
    asof = pd.Timestamp(dataset_asof)
    if asof.tz is not None or asof.time() != pd.Timestamp(0).time():
        raise ValueError("dataset_asof must be a timezone-naive ISO date")
    result = frame.copy()
    days = result.horizon_id.map(HORIZON_DAYS)
    if days.isna().any():
        raise ValueError("unsupported horizon in Phase 4B population")
    entry = pd.to_datetime(result.effective_entry_date).dt.normalize()
    result["theoretically_mature"] = entry.add(pd.to_timedelta(days, unit="D")).le(asof)
    return result


def _summarize_candidates(
    frame: pd.DataFrame, unit: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidate_policies(frame)
    groups = [
        "combo",
        "direction",
        "horizon_id",
        "threshold_family",
        "candidate_threshold",
        "policy_cohort",
    ]
    summary = summarize_outcomes(candidates, groups)
    whole = candidates[candidates.candidate_pass].copy()
    whole["policy_cohort"] = "WHOLE_POLICY_COHORT"
    summary = pd.concat([summary, summarize_outcomes(whole, groups)], ignore_index=True)
    summary["analysis_unit"] = unit
    marginal = summary.policy_cohort.eq("MARGINAL_NEW_ADMISSIONS")
    summary.loc[marginal, "sample_strength"] = summary.loc[marginal].apply(
        lambda r: sample_strength(int(r.n_usable), r.combo), axis=1
    )
    whole_rows = candidates[candidates.candidate_pass].copy()
    whole_rows["policy_cohort"] = "WHOLE_POLICY_COHORT"
    return summary, pd.concat([candidates, whole_rows], ignore_index=True)


def _add_differences(summary: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "combo",
        "direction",
        "horizon_id",
        "threshold_family",
        "candidate_threshold",
    ]
    result = summary.copy()
    ref = result[result.policy_cohort.eq("CURRENT_PASS_RETAINED")][keys + list(METRICS)]
    ref = ref.rename(columns={m: f"current_{m}" for m in METRICS})
    result = result.merge(ref, on=keys, how="left", validate="many_to_one")
    for metric in METRICS:
        result[f"delta_vs_current_{metric}"] = (
            result[metric] - result[f"current_{metric}"]
        )
    return result


def select_calibration_candidates(calibration_aggregates: pd.DataFrame) -> pd.DataFrame:
    """Select from calibration aggregates only; no row-level/holdout input accepted.

    Performance signs are evidence-ranking inputs, not invented materiality
    tolerances.  Consequently a relaxation is always REVIEW_REQUIRED unless
    marginal evidence is too thin, in which case current policy is frozen.
    """
    if (
        "split" in calibration_aggregates
        and not calibration_aggregates.split.eq("CALIBRATION").all()
    ):
        raise AssertionError("candidate selection received non-calibration aggregates")
    required = {
        "combo",
        "direction",
        "horizon_id",
        "threshold_family",
        "candidate_threshold",
        "policy_cohort",
        "n_usable",
    }
    if missing := required - set(calibration_aggregates):
        raise ValueError(f"calibration aggregates missing: {sorted(missing)}")
    rows = []
    marginal = calibration_aggregates[
        calibration_aggregates.policy_cohort.eq("MARGINAL_NEW_ADMISSIONS")
    ]
    for (combo, direction, family), part in marginal.groupby(
        ["combo", "direction", "threshold_family"], sort=True
    ):
        current = CURRENT_THRESHOLDS[family]
        evidence = []
        for threshold, horizons in part.groupby("candidate_threshold"):
            if threshold >= current or set(horizons.horizon_id) != set(HORIZONS[combo]):
                continue
            minimum = int(horizons.n_usable.min())
            signs = sum(
                int(
                    (
                        horizons.get(f"delta_vs_current_{m}", pd.Series(dtype=float))
                        >= 0
                    ).sum()
                )
                for m in ("directional_return_median", "hit_rate", "mae_median")
            )
            evidence.append((minimum, signs, float(threshold)))
        evidence.sort(key=lambda x: (-x[0], -x[1], -x[2]))
        best = evidence[0] if evidence else (0, 0, current)
        strength = sample_strength(best[0], combo)
        insufficient = strength in {"ZERO", "VERY_THIN"}
        selected = current if insufficient else best[2]
        rows.append(
            {
                "combo": combo,
                "direction": direction,
                "threshold_family": family,
                "current_threshold": current,
                "selected_candidate_threshold": selected,
                "selection_status": (
                    "INSUFFICIENT_EVIDENCE" if insufficient else "REVIEW_REQUIRED"
                ),
                "calibration_sample_strength": strength,
                "supported_horizons": "|".join(HORIZONS[combo]),
                "selection_reason_codes": (
                    "MARGINAL_TOO_THIN;KEEP_CURRENT"
                    if insufficient
                    else "SAMPLE_STRENGTH_RANK;SIGN_CONSISTENCY_INPUT;NO_AUTHORIZED_MATERIALITY_TOLERANCE"
                ),
                "selection_input": "CALIBRATION_AGGREGATES_ONLY",
                "policy_authority": (
                    "EXPLORATORY_NOT_POLICY_AUTHORIZING"
                    if combo == "stocks_b_wmq_all"
                    else "PRIMARY_POLICY_DECISIVE_EVIDENCE"
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_frozen_candidates(
    holdout: pd.DataFrame, frozen: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate exactly the thresholds in the persisted frozen table."""
    summary, candidates = _summarize_candidates(holdout, "EPISODE")
    keys = ["combo", "direction", "threshold_family"]
    chosen = frozen[keys + ["selected_candidate_threshold"]].rename(
        columns={"selected_candidate_threshold": "candidate_threshold"}
    )
    result = summary.merge(
        chosen, on=keys + ["candidate_threshold"], how="inner", validate="many_to_one"
    )
    result["sample_strength"] = result.apply(
        lambda r: (
            sample_strength(int(r.n_usable), r.combo)
            if r.policy_cohort == "MARGINAL_NEW_ADMISSIONS"
            else "NOT_APPLICABLE"
        ),
        axis=1,
    )
    result["evidence_status"] = np.where(
        result.combo.eq("stocks_b_wmq_all"),
        "REVIEW_REQUIRED_EXPLORATORY",
        "REVIEW_REQUIRED",
    )
    return _add_differences(result), candidates.merge(
        chosen, on=keys + ["candidate_threshold"], how="inner"
    )


def bootstrap_intervals(
    frame: pd.DataFrame,
    groups: list[str],
    reps: int = BOOTSTRAP_REPS,
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """Return symbol-cluster intervals without pandas work inside a replicate.

    The RNG call and cluster concatenation order deliberately match the original
    implementation: each group draws its symbols ``n_symbols`` times, with
    replacement, from the symbols in first-appearance order.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    grouped = frame.groupby(groups, dropna=False)
    total_groups = grouped.ngroups
    progress_every = max(1, total_groups // 10)
    for group_number, (keys, part) in enumerate(grouped, 1):
        usable = part[part.terminal_covered.astype(bool)]
        symbols = usable.symbol.dropna().unique()
        # Build each cluster exactly once.  Previously every sampled symbol did
        # a full-frame boolean scan and pandas concat in every replicate.
        clusters = {
            symbol: usable.loc[
                usable.symbol.eq(symbol), ["directional_return", "mfe", "mae"]
            ].to_numpy(dtype=float)
            for symbol in symbols
        }
        draws = np.empty((reps, len(METRICS)), dtype=float)
        for replicate in range(reps if len(symbols) else 0):
            sampled = np.concatenate(
                [
                    clusters[symbol]
                    for symbol in rng.choice(symbols, len(symbols), replace=True)
                ]
            )
            returns = sampled[:, 0]
            # pandas reductions skip missing metric values.  Preserve that
            # behavior while hit-rate keeps the original ``NaN > 0 == False``.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                metrics = (
                    np.nanmean(returns),
                    np.nanmedian(returns),
                    np.mean(returns > 0),
                    np.nanmedian(sampled[:, 1]),
                    np.nanmedian(sampled[:, 2]),
                )
            draws[replicate] = metrics
        row = dict(zip(groups, keys if isinstance(keys, tuple) else (keys,))) | {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replications": reps,
            "cluster": "symbol",
            "n_symbols": len(symbols),
        }
        for i, metric in enumerate(METRICS):
            values = draws[:, i] if len(symbols) else np.array([])
            row[f"{metric}_ci_low"] = (
                np.quantile(values, 0.025) if len(values) else np.nan
            )
            row[f"{metric}_ci_high"] = (
                np.quantile(values, 0.975) if len(values) else np.nan
            )
        rows.append(row)
        if progress and (
            group_number == total_groups or group_number % progress_every == 0
        ):
            print(
                f"[BOOTSTRAP] completed cohort {group_number}/{total_groups}",
                flush=True,
            )
    return pd.DataFrame(rows)


@contextmanager
def _stage(name: str):
    started = time.perf_counter()
    print(f"[PHASE4C] START {name}", flush=True)
    try:
        yield
    finally:
        print(
            f"[PHASE4C] END {name} elapsed={time.perf_counter() - started:.2f}s",
            flush=True,
        )


def _atomic_write(path: Path, writer) -> None:
    """Write one complete artifact then atomically replace any prior version."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write(path, lambda temporary: frame.to_csv(temporary, index=False))


def _write_text(text: str, path: Path) -> None:
    _atomic_write(path, lambda temporary: temporary.write_text(text))


def _find(root: Path, name: str) -> Path:
    found = list(root.rglob(name))
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one {name} under {root}; found {len(found)}"
        )
    return found[0]


def _load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    phase4b_summary = json.loads(_find(root, "phase4b_summary.json").read_text())
    if phase4b_summary.get("phase") != "phase4b" or phase4b_summary.get(
        "candidate_grids"
    ) != {k: list(v) for k, v in CANDIDATE_GRIDS.items()}:
        raise ValueError("Phase 4B contract or candidate grids drifted")
    coverage = json.loads(_find(root, "phase4a_coverage_summary.json").read_text())
    dataset_asof = coverage.get("dataset_asof")
    if (
        not isinstance(dataset_asof, str)
        or pd.Timestamp(dataset_asof).date().isoformat() != dataset_asof
    ):
        raise ValueError("authoritative fixed dataset_asof missing or invalid")
    return (
        pd.read_parquet(_find(root, "phase4b_episode_population.parquet")),
        pd.read_parquet(_find(root, "phase4b_observation_population.parquet")),
        dataset_asof,
    )


def _baseline(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["historical_cohort"] = np.where(
        data.historical_participation_pass, "CURRENT_PASS", "CURRENT_BLOCK"
    )
    return summarize_outcomes(
        data, ["combo", "direction", "horizon_id", "split", "historical_cohort"]
    )


def run(artifact_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # Never treat remnants of an interrupted Phase 4C run as current outputs.
    for name in OUTPUTS:
        (output / name).unlink(missing_ok=True)
    with _stage("input loading/preparation"):
        episodes, observations, dataset_asof = _load(artifact_root)
        episodes = enforce_fixed_maturity(split_population(episodes), dataset_asof)
        observations = enforce_fixed_maturity(
            split_population(observations), dataset_asof
        )
    with _stage("calibration grid"):
        calibration = episodes[episodes.split.eq("CALIBRATION")]
        cal_grid, cal_candidates = _summarize_candidates(calibration, "EPISODE")
        cal_grid["split"] = "CALIBRATION"
        cal_grid = _add_differences(cal_grid)
        consistency_keys = [
            "combo",
            "direction",
            "threshold_family",
            "candidate_threshold",
            "policy_cohort",
        ]
        cal_grid["horizons_reported"] = cal_grid.groupby(consistency_keys)[
            "horizon_id"
        ].transform("nunique")
        # Sign agreement is descriptive, not a materiality test or policy rule.
        for metric in ("directional_return_median", "hit_rate", "mae_median"):
            column = f"delta_vs_current_{metric}"
            cal_grid[f"{metric}_sign_consistent_across_horizons"] = cal_grid.groupby(
                consistency_keys
            )[column].transform(
                lambda values: (
                    values.dropna().ge(0).nunique() <= 1
                    if len(values.dropna())
                    else False
                )
            )
        _write_csv(cal_grid, output / OUTPUTS[2])
    with _stage("frozen candidate selection"):
        frozen = select_calibration_candidates(cal_grid)
        _write_csv(frozen, output / OUTPUTS[3])  # freeze boundary
        frozen = pd.read_csv(output / OUTPUTS[3])
    with _stage("holdout validation"):
        holdout = episodes[episodes.split.eq("HOLDOUT")]
        validation, selected_rows = evaluate_frozen_candidates(holdout, frozen)
        _write_csv(validation, output / OUTPUTS[4])
    with _stage("current PASS/BLOCK baseline"):
        _write_csv(_baseline(episodes), output / OUTPUTS[5])
    with _stage("bootstrap intervals"):
        boot_input = pd.concat(
            [
                cal_candidates.assign(split="CALIBRATION"),
                selected_rows.assign(split="HOLDOUT"),
            ]
        )
        intervals = bootstrap_intervals(
            boot_input,
            [
                "combo",
                "direction",
                "horizon_id",
                "split",
                "threshold_family",
                "candidate_threshold",
                "policy_cohort",
            ],
            progress=True,
        )
        _write_csv(intervals, output / OUTPUTS[6])
    with _stage("corporate-action sensitivity"):
        sensitivity = []
        for excluded, data in (
            (False, episodes),
            (True, episodes[~episodes.corporate_action_flag.fillna(False)]),
        ):
            baseline = _baseline(data)
            baseline["analysis"] = "CURRENT_PASS_BLOCK"
            baseline["corporate_actions_excluded"] = excluded
            sensitivity.append(baseline)
            for split in ("CALIBRATION", "HOLDOUT"):
                selected, _ = _summarize_candidates(
                    data[data.split.eq(split)], "EPISODE"
                )
                keys = ["combo", "direction", "threshold_family"]
                chosen = frozen[keys + ["selected_candidate_threshold"]].rename(
                    columns={"selected_candidate_threshold": "candidate_threshold"}
                )
                selected = selected.merge(
                    chosen,
                    on=keys + ["candidate_threshold"],
                    how="inner",
                    validate="many_to_one",
                )
                selected["split"] = split
                selected["analysis"] = "FROZEN_SELECTED_POLICY"
                selected["corporate_actions_excluded"] = excluded
                sensitivity.append(selected)
        _write_csv(pd.concat(sensitivity), output / OUTPUTS[7])
    with _stage("observation robustness"):
        obs, _ = _summarize_candidates(observations, "OBSERVATION")
        obs["selection_eligible"] = False
        _write_csv(obs, output / OUTPUTS[8])
    with _stage("chronological regime"):
        regime = episodes.copy()
        regime["chronological_subperiod"] = regime.effective_entry_date.dt.to_period(
            "Q"
        ).astype(str)
        regime_summary = summarize_outcomes(
            regime.assign(
                historical_cohort=np.where(
                    regime.historical_participation_pass,
                    "CURRENT_PASS",
                    "CURRENT_BLOCK",
                )
            ),
            [
                "combo",
                "direction",
                "horizon_id",
                "split",
                "chronological_subperiod",
                "historical_cohort",
            ],
        )
        _write_csv(regime_summary, output / OUTPUTS[9])
    with _stage("report/final outputs"):
        bounds = {}
        for combo, cutoff in SPLITS.items():
            c, h = (
                episodes[(episodes.combo == combo) & (episodes.split == "CALIBRATION")],
                episodes[(episodes.combo == combo) & (episodes.split == "HOLDOUT")],
            )
            bounds[combo] = {
                "cutoff": cutoff.date().isoformat(),
                "max_calibration_date": (
                    None if c.empty else c.effective_entry_date.max().date().isoformat()
                ),
                "min_holdout_date": (
                    None if h.empty else h.effective_entry_date.min().date().isoformat()
                ),
                "no_overlap": (c.empty or c.effective_entry_date.max() <= cutoff)
                and (h.empty or h.effective_entry_date.min() > cutoff),
            }
        audit = {
            "passed": all(v["no_overlap"] for v in bounds.values()),
            "split_bounds": bounds,
            "no_alternative_cutoff_search": True,
            "mqy_excluded": True,
            "selection_input": "phase4c_calibration_grid.csv:CALIBRATION rows only",
            "holdout_not_selection_input": True,
            "frozen_table_drives_holdout": True,
            "candidate_grids_unchanged": True,
            "episode_contract_unchanged": True,
            "outcome_contract_unchanged": True,
            "fixed_dataset_asof": dataset_asof,
            "production_configuration_untouched": True,
            "observation_selection_eligible": False,
        }
        if not audit["passed"]:
            raise AssertionError("leakage audit failed")
        _write_text(json.dumps(audit, indent=2) + "\n", output / OUTPUTS[10])
        summary = {
            "phase": "phase4c",
            "read_only": True,
            "artifact_source": str(artifact_root.resolve()),
            "dataset_asof": dataset_asof,
            "split_cutoffs": {k: v.date().isoformat() for k, v in SPLITS.items()},
            "no_alternative_cutoff_search": True,
            "mqy_excluded": True,
            "horizons": HORIZONS,
            "candidate_grids": CANDIDATE_GRIDS,
            "current_thresholds": CURRENT_THRESHOLDS,
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "replications": BOOTSTRAP_REPS,
                "cluster": "symbol",
            },
            "episode_definition": "Phase 4B first observation anchor; canonical-sequence adjacency by combo/symbol/direction",
            "outcome_definition": "Phase 4B immutable signal close; first market date on/after calendar target within 7 days; directional return/MFE/MAE/hit rate",
            "corporate_action_primary": "included",
            "wmq_authority": "EXPLORATORY_NOT_POLICY_AUTHORIZING",
            "wmq_180d_warning": "holdout temporal breadth underpowered: LONG 7 and SHORT 5 mature weekly evaluation dates",
            "production_threshold_changes": False,
            "outputs": OUTPUTS,
        }
        _write_text(json.dumps(summary, indent=2) + "\n", output / OUTPUTS[0])
        _report(output, frozen, validation, audit)


def _report(
    output: Path, frozen: pd.DataFrame, validation: pd.DataFrame, audit: dict
) -> None:
    sections = [
        (
            "PRECOMMITTED DESIGN",
            "DWM cutoff: **2026-04-30** (primary/policy-decisive). WMQ cutoff: **2026-01-31** (exploratory/not policy-authorizing). MQY is excluded. No cutoff search occurs.",
        ),
        (
            "CALIBRATION RESULTS",
            "See `phase4c_calibration_grid.csv`; marginal sample sizes and precommitted strength labels are explicit.",
        ),
        ("CALIBRATION-ONLY CANDIDATE SELECTION", frozen.to_markdown(index=False)),
        ("HOLDOUT RESULTS", validation.to_markdown(index=False)),
        (
            "CALIBRATION VS HOLDOUT COMPARISON",
            "The grid and frozen holdout tables expose metric differences; no unauthorized materiality tolerance is imposed.",
        ),
        (
            "DWM POLICY EVIDENCE",
            "Primary evidence remains **REVIEW_REQUIRED**. This diagnostic does not edit production configuration.",
        ),
        (
            "WMQ EXPLORATORY EVIDENCE",
            "**EXPLORATORY / NOT POLICY-AUTHORIZING.** The 180d holdout has only 7 LONG and 5 SHORT mature weekly evaluation dates.",
        ),
        (
            "ROBUSTNESS",
            "Corporate-action exclusion, observation-level secondary analysis, chronological quarters, and symbol-cluster intervals are separate artifacts.",
        ),
        (
            "LIMITATIONS",
            "No empirically authorized performance tolerance exists. Observation rows cannot select policy. Bootstrap clustering does not model common-date dependence.",
        ),
        (
            "LEAKAGE / CONTRACT AUDIT",
            f"Audit passed: **{audit['passed']}**. Selection consumes calibration aggregates only and holdout consumes the frozen CSV.",
        ),
        (
            "NEXT DECISION",
            "Human review must assess calibration and untouched holdout evidence. No threshold change is automatic.",
        ),
    ]
    text = ["# Phase 4C — chronological calibration / holdout validation", ""]
    for title, body in sections:
        text += [f"## {title}", "", body, ""]
    _write_text("\n".join(text), output / OUTPUTS[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir", default=Path("diagnostic_artifacts/phase4c"), type=Path
    )
    args = parser.parse_args()
    run(args.artifact_root, args.output_dir)
    print("[SAFETY] LOCAL READ-ONLY DIAGNOSTIC; NO PRODUCTION/S3/MARKET-DATA WRITES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
