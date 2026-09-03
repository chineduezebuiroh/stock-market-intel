#!/usr/bin/env python3
"""Bounded, read-only cross-combo Phase 3B characterization.

Consumes only the locally validated outputs produced by the history validator.
No storage client is imported here: this stage cannot write to S3.
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
    construct_episodes,
    scenario_pass,
    transition_table,
)

COMBOS = tuple(COMBO_SPECS)
LIMITED_HISTORY_LABEL = "LIMITED HISTORY — 8 canonical market dates"
GUARDS = {
    "stocks_c_dwm_all": (177, 399_851, "2025-12-30T00:10:19Z", "2026-09-02T02:38:57Z"),
    "stocks_b_wmq_all": (39, 82_880, "2026-01-03T12:01:33Z", "2026-08-29T12:24:57Z"),
    "stocks_a_mqy_all": (14, 19_408, "2026-01-04T12:07:53Z", "2026-08-09T15:51:28Z"),
}
STRONG_AVAILABLE = (0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10)
STRONG_UNAVAILABLE = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
MODERATE = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30)


def validate_population(combo: str, frame: pd.DataFrame, coverage: dict) -> None:
    """Apply pinned-boundary, append-only, exact-validation fail-closed guards."""
    min_artifacts, min_rows, first, last = GUARDS[combo]
    errors = []
    if coverage.get("combo") != combo:
        errors.append("combo mismatch")
    if coverage.get("supported_artifact_count", 0) < min_artifacts:
        errors.append("supported artifact count regressed")
    if len(frame) < min_rows:
        errors.append("canonical observation count regressed")
    if pd.Timestamp(
        coverage.get("first_supported_five_component_artifact")
    ) != pd.Timestamp(first):
        errors.append("first supported boundary changed")
    if pd.Timestamp(
        coverage.get("last_supported_five_component_artifact")
    ) < pd.Timestamp(last):
        errors.append("last supported artifact regressed")
    if coverage.get("validation_error_count") != 0:
        errors.append("schema/scoring validation errors are present")
    if coverage.get("scoring_contract") != "five_component_with_participation":
        errors.append("scoring contract drift")
    if not coverage.get("strict_schema") or not coverage.get(
        "supported_era_contiguous"
    ):
        errors.append("strict/contiguous modern era validation failed")
    if frame.duplicated(["symbol", "lower_date"]).any():
        errors.append("canonical uniqueness failed")
    if errors:
        raise AssertionError(f"{combo} Phase 3B guard failed: {', '.join(errors)}")


def _distribution(frame: pd.DataFrame, value: str, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.dropna(subset=[value]).groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = part[value]
        rows.append(
            dict(zip(groups, keys))
            | {
                "count": len(values),
                "mean": values.mean(),
                "p25": values.quantile(0.25),
                "p50": values.quantile(0.5),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.9),
                "p95": values.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def characterize(
    frames: dict[str, pd.DataFrame], output: Path
) -> dict[str, pd.DataFrame]:
    """Build combo/direction tables from validated canonical observations."""
    output.mkdir(parents=True, exist_ok=True)
    directional = []
    episodes = []
    transitions = []
    for combo, canonical in frames.items():
        d = build_directional_opportunities(canonical, combo)
        ep, ids = construct_episodes(d)
        d["episode_id"] = ids
        ep["combo"] = combo
        tr = transition_table(d)
        tr["combo"] = combo
        directional.append(d)
        episodes.append(ep)
        transitions.append(tr)
    data = pd.concat(directional, ignore_index=True)
    preserved = [
        "combo",
        "direction",
        "symbol",
        "lower_date",
        "artifact_execution_utc",
        "logic_era",
        "source_s3_key",
        "pre_participation",
        "admitted_five_component",
        "participation_only_blocker",
        "route_class",
    ]
    data.reindex(columns=preserved).to_parquet(
        output / "supported_directional_observations.parquet", index=False
    )
    pre = data[data.pre_participation].copy()

    population = (
        pre.groupby(["combo", "direction"])
        .agg(
            preparticipation_count=("symbol", "size"),
            admitted_count=("admitted_five_component", "sum"),
            blocker_count=("participation_only_blocker", "sum"),
            unique_symbols=("symbol", "nunique"),
        )
        .reset_index()
    )
    totals = (
        data.groupby(["combo", "direction"])
        .size()
        .rename("canonical_observations")
        .reset_index()
    )
    population = totals.merge(population)
    population["pass_rate"] = (
        population.admitted_count / population.preparticipation_count
    )
    population["blocker_rate"] = (
        population.blocker_count / population.preparticipation_count
    )

    route = (
        pre.groupby(["combo", "direction", "route_class"])
        .size()
        .rename("count")
        .reset_index()
    )
    route["share"] = route["count"] / route.groupby(["combo", "direction"])[
        "count"
    ].transform("sum")
    route_wide = route.pivot_table(
        index=["combo", "direction"],
        columns="route_class",
        values="count",
        fill_value=0,
    ).reset_index()
    for name in ("LOWER_ONLY", "MIDDLE_ONLY", "BOTH", "NEITHER"):
        if name not in route_wide:
            route_wide[name] = 0
    for name in ("LOWER_ONLY", "MIDDLE_ONLY", "BOTH", "NEITHER"):
        route_wide[f"{name}_share"] = route_wide[name] / route_wide[
            ["LOWER_ONLY", "MIDDLE_ONLY", "BOTH", "NEITHER"]
        ].sum(axis=1)
    for side in ("lower", "middle"):
        route_wide[f"{side}_route_eligible_count"] = (
            pre.groupby(["combo", "direction"])[f"{side}_threshold"]
            .apply(lambda x: x.notna().sum())
            .values
        )
        route_wide[f"{side}_route_pass_rate"] = (
            pre.groupby(["combo", "direction"])[f"{side}_participation_pass"]
            .mean()
            .values
        )

    routes = []
    for side in ("lower", "middle"):
        x = pre[
            [
                "combo",
                "direction",
                "symbol",
                "episode_id",
                "overall_participation_pass",
                "upper_has_wyckoff",
                f"{side}_sigvol_tier",
                f"{side}_ratio",
                f"{side}_threshold",
                f"{side}_normalized_ratio",
                f"{side}_margin",
                f"{side}_participation_pass",
            ]
        ].copy()
        x.columns = [c.replace(f"{side}_", "") for c in x.columns]
        x["route"] = side
        x["tier_branch"] = np.select(
            [
                x.sigvol_tier.eq(1),
                x.sigvol_tier.eq(2) & x.upper_has_wyckoff,
                x.sigvol_tier.eq(2),
            ],
            ["MODERATE", "STRONG_UPPER_AVAILABLE", "STRONG_UPPER_UNAVAILABLE"],
            default="TIER_0",
        )
        routes.append(x)
    routes = pd.concat(routes, ignore_index=True)
    eligible = routes[routes.sigvol_tier.isin([1, 2])]
    ratios = _distribution(
        eligible, "ratio", ["combo", "direction", "route", "tier_branch"]
    )
    normalized = _distribution(
        eligible, "normalized_ratio", ["combo", "direction", "tier_branch"]
    )
    tier = ratios.merge(
        eligible.groupby(["combo", "direction", "route", "tier_branch"])
        .agg(
            route_pass_rate=("participation_pass", "mean"),
            opportunity_participation_pass_rate=("overall_participation_pass", "mean"),
        )
        .reset_index()
    )
    distance = eligible.copy()
    distance["threshold_bucket"] = pd.cut(
        distance.normalized_ratio,
        [-np.inf, 0.25, 0.5, 0.75, 0.9, 1, np.inf],
        right=False,
        labels=[
            "<0.25x",
            "0.25–0.50x",
            "0.50–0.75x",
            "0.75–0.90x",
            "0.90–1.00x",
            ">=1.00x",
        ],
    )
    distance = (
        distance.groupby(
            ["combo", "direction", "route", "tier_branch", "threshold_bucket"],
            observed=False,
        )
        .agg(
            count=("symbol", "size"),
            normalized_median=("normalized_ratio", "median"),
            margin_median=("margin", "median"),
        )
        .reset_index()
    )

    one_d = []
    families = (
        ("STRONG_UPPER_AVAILABLE", STRONG_AVAILABLE, "strong_available"),
        ("STRONG_UPPER_UNAVAILABLE", STRONG_UNAVAILABLE, "strong_unavailable"),
        ("MODERATE", MODERATE, "moderate"),
    )
    for (combo, direction), part in pre.groupby(["combo", "direction"]):
        current = part.overall_participation_pass
        current_n = int(current.sum())
        for family, candidates, argument in families:
            for threshold in candidates:
                kwargs = {argument: threshold}
                passed = scenario_pass(part, **kwargs)
                newly = passed & ~current
                one_d.append(
                    {
                        "combo": combo,
                        "direction": direction,
                        "threshold_family": family,
                        "threshold": threshold,
                        "admitted_count": int(passed.sum()),
                        "incremental_admissions_vs_current": int(
                            passed.sum() - current_n
                        ),
                        "incremental_pct_current_admitted": (
                            (passed.sum() - current_n) / current_n
                            if current_n
                            else np.nan
                        ),
                        "incremental_pct_preparticipation": (passed.sum() - current_n)
                        / len(part),
                        "newly_admitted_unique_symbols": part.loc[
                            newly, "symbol"
                        ].nunique(),
                        "newly_admitted_episodes": part.loc[
                            newly, "episode_id"
                        ].nunique(),
                    }
                )
    one_d = pd.DataFrame(one_d)
    two_d = []
    for (combo, direction), part in pre.groupby(["combo", "direction"]):
        current = int(part.overall_participation_pass.sum())
        for strong in STRONG_AVAILABLE:
            for moderate in MODERATE:
                passed = scenario_pass(part, strong_available=strong, moderate=moderate)
                two_d.append(
                    {
                        "combo": combo,
                        "direction": direction,
                        "strong_upper_available_threshold": strong,
                        "moderate_threshold": moderate,
                        "strong_upper_unavailable_threshold": 0.10,
                        "admitted_count": int(passed.sum()),
                        "incremental_admissions_vs_current": int(
                            passed.sum() - current
                        ),
                    }
                )
    temporal = []
    for (combo, direction), part in pre.groupby(["combo", "direction"]):
        key = (
            part.lower_date.astype(str)
            if combo == "stocks_a_mqy_all"
            else pd.to_datetime(part.lower_date).dt.to_period("M").astype(str)
        )
        for period, group in part.groupby(key):
            temporal.append(
                {
                    "combo": combo,
                    "direction": direction,
                    "period": period,
                    "preparticipation_count": len(group),
                    "pass_rate": group.overall_participation_pass.mean(),
                    "blocker_rate": group.participation_only_blocker.mean(),
                    "history_note": (
                        LIMITED_HISTORY_LABEL if combo == "stocks_a_mqy_all" else ""
                    ),
                }
            )
    concentration = []
    for (combo, direction), part in pre.groupby(["combo", "direction"]):
        for population_name, mask in (
            ("ADMITTED", part.admitted_five_component),
            ("BLOCKER", part.participation_only_blocker),
        ):
            counts = part.loc[mask].symbol.value_counts()
            total = counts.sum()
            for n in (10, 25, 50, 100):
                concentration.append(
                    {
                        "combo": combo,
                        "direction": direction,
                        "population": population_name,
                        "top_n": n,
                        "share": counts.head(n).sum() / total if total else np.nan,
                    }
                )
    tables = {
        "population_summary": population,
        "route_outcomes": route_wide,
        "sigvol_tier_comparison": tier,
        "ratio_distributions": ratios,
        "normalized_threshold_distributions": normalized,
        "distance_to_threshold": distance,
        "threshold_sensitivity_1d": one_d,
        "threshold_sensitivity_2d": pd.DataFrame(two_d),
        "temporal_summary": pd.DataFrame(temporal),
        "symbol_concentration": pd.DataFrame(concentration),
        "episode_summary": pd.concat(episodes, ignore_index=True),
        "transition_summary": pd.concat(transitions, ignore_index=True),
    }
    for name, table in tables.items():
        table.to_csv(output / f"cross_combo_{name}.csv", index=False)
    for combo in COMBOS:
        combo_dir = output / combo
        combo_dir.mkdir(exist_ok=True)
        for name, table in tables.items():
            if "combo" in table:
                table[table.combo.eq(combo)].to_csv(
                    combo_dir / f"{name}.csv", index=False
                )
    return tables


def write_report(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    pop = tables["population_summary"]
    columns = list(pop.columns)
    markdown = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    markdown.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in pop.itertuples(index=False, name=None)
    )
    lines = [
        "# Cross-combo Phase 3B participation characterization",
        "",
        "> Read-only descriptive policy characterization. No outcomes, optimization, or production threshold recommendation.",
        "",
        "## Admission comparison",
        "",
        "\n".join(markdown),
        "",
        "## Required interpretation questions",
        "",
        "1. Admission and blocker percentages for each combo/direction are reported in the table above.",
        "2. Moderate 25% restrictiveness must be compared using the normalized distributions and sensitivity tables; divergence identifies a Phase 4 test family, not a change.",
        "3. Strong 5% restrictiveness is likewise reported separately by combo and direction.",
        "4. The 10% no-upper branch is isolated as `STRONG_UPPER_UNAVAILABLE` in all distribution and sensitivity outputs.",
        "5. Lower/middle use is reported as lower-only, middle-only, both, and neither in route outcomes.",
        "6. Top-10/25/50/100 admitted and blocker shares quantify symbol concentration.",
        "7. Episode and transition tables distinguish short confirmation delays from persistent blockers.",
        "8. Threshold families with the largest cross-combo descriptive divergence deserve Phase 4 outcome testing; this report does not select or recommend a production threshold.",
        "",
        f"## M/Q/Y temporal guardrail\n\n**{LIMITED_HISTORY_LABEL}.** No long-run stability conclusion may be drawn.",
    ]
    (output / "phase3b_report.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    frames = {}
    coverages = {}
    for combo in COMBOS:
        root = args.validated_root / combo
        frame = pd.read_parquet(root / "supported_observations.parquet")
        coverage = json.loads((root / "coverage_summary.json").read_text())
        validate_population(combo, frame, coverage)
        frames[combo] = frame
        coverages[combo] = coverage
    tables = characterize(frames, args.output_dir)
    write_report(args.output_dir, tables)
    summary = {
        "phase": "phase3b",
        "validation_scope": "cross_combo_phase3b",
        "read_only": True,
        "combos": coverages,
        "mqy_temporal_guardrail": LIMITED_HISTORY_LABEL,
        "production_threshold_changes": False,
    }
    (args.output_dir / "phase3b_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
