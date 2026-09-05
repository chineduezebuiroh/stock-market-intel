import inspect

import pandas as pd
from pandas.testing import assert_frame_equal

import diagnostics.analyze_stock_options_phase4c as phase4c
from diagnostics.analyze_stock_options_phase4b import (
    CANDIDATE_GRIDS,
    construct_preparticipation_episodes,
)


def _population():
    rows = []
    for combo, dates, horizon in (
        ("stocks_c_dwm_all", ("2026-04-30", "2026-05-01"), "DWM_1x"),
        ("stocks_b_wmq_all", ("2026-01-31", "2026-02-01"), "WMQ_1x"),
        ("stocks_a_mqy_all", ("2026-01-01", "2026-02-01"), "MQY_1x"),
    ):
        for date in dates:
            rows.append(
                {
                    "combo": combo,
                    "direction": "LONG",
                    "symbol": "A",
                    "horizon_id": horizon,
                    "effective_entry_date": date,
                }
            )
    return pd.DataFrame(rows)


def test_exact_precommitted_splits_no_overlap_and_mqy_excluded():
    assert phase4c.SPLITS == {
        "stocks_c_dwm_all": pd.Timestamp("2026-04-30"),
        "stocks_b_wmq_all": pd.Timestamp("2026-01-31"),
    }
    result = phase4c.split_population(_population())
    assert not result.combo.eq("stocks_a_mqy_all").any()
    for combo, cutoff in phase4c.SPLITS.items():
        cal = result[(result.combo == combo) & (result.split == "CALIBRATION")]
        hold = result[(result.combo == combo) & (result.split == "HOLDOUT")]
        assert cal.effective_entry_date.max() <= cutoff
        assert hold.effective_entry_date.min() > cutoff
        assert set(cal.index).isdisjoint(hold.index)


def _aggregates():
    rows = []
    for horizon in phase4c.HORIZONS["stocks_c_dwm_all"]:
        for threshold, n, delta in ((0.15, 60, 0.1), (0.20, 30, -0.1)):
            rows.append(
                {
                    "combo": "stocks_c_dwm_all",
                    "direction": "LONG",
                    "horizon_id": horizon,
                    "threshold_family": "MODERATE",
                    "candidate_threshold": threshold,
                    "policy_cohort": "MARGINAL_NEW_ADMISSIONS",
                    "n_usable": n,
                    "split": "CALIBRATION",
                    "delta_vs_current_directional_return_median": delta,
                    "delta_vs_current_hit_rate": delta,
                    "delta_vs_current_mae_median": delta,
                }
            )
    return pd.DataFrame(rows)


def test_exact_grids_sample_labels_and_calibration_only_selection():
    assert CANDIDATE_GRIDS["MODERATE"] == (
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
    )
    assert phase4c.sample_strength(100, "stocks_c_dwm_all") == "ROBUST"
    assert phase4c.sample_strength(49, "stocks_c_dwm_all") == "THIN"
    assert phase4c.sample_strength(30, "stocks_b_wmq_all") == "MODERATE"
    assert phase4c.sample_strength(0, "stocks_b_wmq_all") == "ZERO"
    selected = phase4c.select_calibration_candidates(_aggregates())
    assert selected.iloc[0].selected_candidate_threshold == 0.15
    assert selected.iloc[0].selection_status == "REVIEW_REQUIRED"
    contaminated = _aggregates().copy()
    contaminated.loc[0, "split"] = "HOLDOUT"
    try:
        phase4c.select_calibration_candidates(contaminated)
    except AssertionError:
        pass
    else:
        raise AssertionError("selection accepted holdout aggregate")


def test_holdout_mutation_cannot_change_selection_and_observations_not_input():
    calibration = _aggregates()
    before = phase4c.select_calibration_candidates(calibration)
    arbitrary_holdout = pd.DataFrame({"directional_return": [999.0]})
    arbitrary_holdout.directional_return *= -999
    after = phase4c.select_calibration_candidates(calibration)
    assert_frame_equal(before, after)
    source = inspect.getsource(phase4c.run)
    assert "select_calibration_candidates(cal_grid)" in source
    assert (
        "observations"
        not in inspect.signature(phase4c.select_calibration_candidates).parameters
    )


def test_frozen_table_drives_holdout_and_wmq_guardrail(monkeypatch):
    frozen = pd.DataFrame(
        {
            "combo": ["stocks_b_wmq_all"],
            "direction": ["LONG"],
            "threshold_family": ["MODERATE"],
            "selected_candidate_threshold": [0.20],
        }
    )
    summary = pd.DataFrame(
        {
            "combo": ["stocks_b_wmq_all"] * 2,
            "direction": ["LONG"] * 2,
            "horizon_id": ["WMQ_1x"] * 2,
            "threshold_family": ["MODERATE"] * 2,
            "candidate_threshold": [0.20, 0.15],
            "policy_cohort": ["MARGINAL_NEW_ADMISSIONS"] * 2,
            "n_usable": [15, 99],
            **{metric: [0.0, 1.0] for metric in phase4c.METRICS},
        }
    )
    candidates = summary.copy()
    monkeypatch.setattr(
        phase4c, "_summarize_candidates", lambda frame, unit: (summary, candidates)
    )
    result, rows = phase4c.evaluate_frozen_candidates(pd.DataFrame(), frozen)
    assert set(result.candidate_threshold) == {0.20}
    assert set(rows.candidate_threshold) == {0.20}
    assert result.evidence_status.eq("REVIEW_REQUIRED_EXPLORATORY").all()


def test_episode_contract_fixed_asof_bootstrap_and_read_only_source():
    canonical = pd.DataFrame(
        {
            "combo": ["stocks_c_dwm_all"] * 3,
            "direction": ["LONG"] * 3,
            "symbol": ["A"] * 3,
            "lower_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "pre_participation": [True, False, True],
        }
    )
    _, anchors = construct_preparticipation_episodes(canonical)
    assert len(anchors) == 2
    assert phase4c.BOOTSTRAP_SEED == 4204
    maturity = phase4c.enforce_fixed_maturity(
        pd.DataFrame(
            {
                "effective_entry_date": ["2026-01-01", "2026-06-01"],
                "horizon_id": ["DWM_1x", "DWM_1x"],
                "theoretically_mature": [False, True],
            }
        ),
        "2026-06-15",
    )
    assert maturity.theoretically_mature.tolist() == [True, False]
    source = inspect.getsource(phase4c)
    assert "phase4a_coverage_summary.json" in source
    assert "corporate_actions_excluded" in source
    assert 'obs["selection_eligible"] = False' in source
    for forbidden in ("put_object", "delete_object", "boto3", "yfinance"):
        assert forbidden not in source


def test_bootstrap_is_deterministic():
    frame = pd.DataFrame(
        {
            "group": ["x"] * 4,
            "symbol": ["A", "A", "B", "B"],
            "terminal_covered": [True] * 4,
            "directional_return": [0.1, -0.1, 0.2, 0.0],
            "mfe": [0.2] * 4,
            "mae": [-0.1] * 4,
        }
    )
    assert_frame_equal(
        phase4c.bootstrap_intervals(frame, ["group"], reps=20),
        phase4c.bootstrap_intervals(frame, ["group"], reps=20),
    )
