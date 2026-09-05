import inspect

import pandas as pd
from pandas.testing import assert_frame_equal

import diagnostics.analyze_phase4c_split_recon as recon


def population(months=12, per_month=60):
    rows = []
    for combo, horizons in recon.SUPPORTED.items():
        for direction in ("LONG", "SHORT"):
            for month in range(months):
                for number in range(per_month):
                    entry = pd.Timestamp("2025-01-01") + pd.DateOffset(
                        months=month, days=number % 20
                    )
                    for horizon in horizons:
                        rows.append(
                            {
                                "combo": combo,
                                "symbol": f"S{number}",
                                "direction": direction,
                                "episode_id": f"{combo}-{direction}-{month}-{number}",
                                "episode_start_date": entry,
                                "effective_entry_date": entry,
                                "episode_end_date": entry,
                                "episode_observation_count": 1,
                                "horizon_id": horizon,
                                "outcome_price_asof": pd.Timestamp("2026-06-30"),
                                "theoretically_mature": True,
                                "terminal_covered": number % 10 != 0,
                                "coverage_status": (
                                    "SYMBOL_HISTORY_STALE_BEFORE_TARGET"
                                    if number % 10 == 0
                                    else "MATURE"
                                ),
                                "historical_participation_pass": number % 2 == 0,
                                "lower_sigvol_tier": 1,
                                "middle_sigvol_tier": 0,
                                "lower_ratio": 0.21,
                                "middle_ratio": 0.0,
                                "upper_has_wyckoff": True,
                                "corporate_action_flag": number == 1,
                            }
                        )
    return pd.DataFrame(rows)


def test_exact_prespecified_grids_and_exclusions():
    assert recon.CANDIDATE_GRIDS["MODERATE"] == (
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
    )
    assert recon.CANDIDATE_GRIDS["STRONG_UPPER_AVAILABLE"] == (
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
    )
    assert recon.CANDIDATE_GRIDS["STRONG_UPPER_UNAVAILABLE"] == (
        0.025,
        0.05,
        0.075,
        0.10,
    )
    assert "WMQ_3x" not in recon.SUPPORTED["stocks_b_wmq_all"]
    assert "stocks_a_mqy_all" not in recon.SUPPORTED


def test_performance_columns_cannot_change_results_and_are_excluded():
    base = population(2, 3)
    first = base.assign(directional_return=1.0, mfe=2.0, mae=-1.0)
    second = base.assign(directional_return=-999.0, mfe=-888.0, mae=777.0)
    left = recon.prepare(first)
    right = recon.prepare(second)
    assert not set(recon.FORBIDDEN_COLUMNS) & set(left)
    assert_frame_equal(left, right)
    cuts = {c: recon.candidate_cutoffs(left, c) for c in recon.SUPPORTED}
    assert_frame_equal(
        recon.count_matrices(left, cuts)[0], recon.count_matrices(right, cuts)[0]
    )


def test_entry_boundary_maturity_stale_participation_and_effective_date():
    frame = recon.prepare(population(2, 4))
    # Deliberately decouple episode start: boundary must still use effective entry.
    frame.loc[:, "episode_start_date"] = pd.Timestamp("1999-01-01")
    cutoff = pd.Timestamp("2025-01-31")
    counts, participation, _ = recon.count_matrices(
        frame, {"stocks_c_dwm_all": [cutoff], "stocks_b_wmq_all": []}
    )
    row = counts.query(
        "direction == 'LONG' and horizon == 'DWM_1x' and split_side == 'CALIBRATION'"
    ).iloc[0]
    assert row.total_episode_anchors == 4
    assert row.theoretically_mature_episodes == 4
    assert row.stale_history_censored == 1
    cohorts = participation.query(
        "direction == 'LONG' and horizon == 'DWM_1x' and split_side == 'CALIBRATION'"
    )
    assert set(cohorts.historical_cohort) == {"CURRENT_PASS", "CURRENT_BLOCK"}


def test_candidate_marginals_and_count_tiers_are_retained():
    frame = recon.prepare(population(2, 6))
    cuts = {"stocks_c_dwm_all": [pd.Timestamp("2025-01-31")], "stocks_b_wmq_all": []}
    result = recon.policy_counts(frame, cuts)
    assert set(result.policy_cohort) == set(recon.COHORTS)
    moderate_20 = result.query(
        "threshold_family == 'MODERATE' and candidate_threshold == .20"
    )
    assert not moderate_20.empty
    assert "MARGINAL_NEW_ADMISSIONS" in set(moderate_20.policy_cohort)
    assert recon.marginal_tier(100, "stocks_c_dwm_all") == "ROBUST_COUNT"
    assert recon.marginal_tier(50, "stocks_b_wmq_all", alternate=True) == "ROBUST_COUNT"


def test_rubrics_all_horizons_latest_and_no_feasible():
    frame = recon.prepare(population(20, 60))
    cuts = {c: recon.candidate_cutoffs(frame, c) for c in recon.SUPPORTED}
    counts, participation, temporal = recon.count_matrices(frame, cuts)
    feasible = recon.feasibility(counts, participation, temporal)
    dwm = feasible.query("combo == 'stocks_c_dwm_all' and direction == 'LONG'")
    assert dwm.preferred_feasible.any()  # 500/500, 100/100 rubric
    wmq = feasible.query("combo == 'stocks_b_wmq_all' and direction == 'LONG'")
    assert wmq.preferred_feasible.any()  # 150/150, 30/30 rubric
    recs = recon.recommendations(feasible)
    assert all("latest_minimum_cutoff" in item for item in recs)
    no = feasible.copy()
    no[["all_horizons_preferred", "all_horizons_minimum"]] = False
    assert all(
        item["minimum_status"] == "NO_FEASIBLE_SPLIT"
        for item in recon.recommendations(no)
    )


def test_longest_horizon_binds_temporal_and_coverage_warnings():
    frame = recon.prepare(population(3, 40))
    # Make the longest horizon thin, and PASS coverage observably unequal.
    longest = frame.horizon_id.isin(["DWM_3x", "WMQ_2x"])
    frame.loc[longest, "terminal_covered"] = False
    frame.loc[frame.historical_participation_pass, "terminal_covered"] = True
    cuts = {c: [pd.Timestamp("2025-02-28")] for c in recon.SUPPORTED}
    counts, participation, temporal = recon.count_matrices(frame, cuts)
    result = recon.feasibility(counts, participation, temporal)
    assert result.coverage_imbalance.any()
    assert temporal.query("combo == 'stocks_c_dwm_all'").short_temporal_span.any()
    assert set(result.direction) == {"LONG", "SHORT"}
    assert not result.groupby(["combo", "direction"]).all_horizons_preferred.any().any()


def test_no_storage_or_external_provider_imports():
    source = inspect.getsource(recon)
    for forbidden in (
        "boto3",
        "yfinance",
        "put_object",
        "delete_object",
        "core.storage",
    ):
        assert forbidden not in source
