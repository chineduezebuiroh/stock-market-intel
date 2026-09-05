import pandas as pd
import inspect
from pandas.testing import assert_frame_equal

from diagnostics.analyze_stock_options_phase4b import (
    CANDIDATE_GRIDS,
    SUPPORTED_HORIZONS,
    candidate_policies,
    construct_preparticipation_episodes,
    summarize_outcomes,
    symbol_cluster_bootstrap,
)
from diagnostics.outcomes.metrics import directional_return
from diagnostics.outcomes.specs import Direction
import diagnostics.analyze_stock_options_phase4b as phase4b


def _canonical():
    rows = []
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-09"])
    for direction in ("LONG", "SHORT"):
        for date, qualified in zip(dates, [True, True, False, True]):
            rows.append(
                {
                    "combo": "stocks_c_dwm_all",
                    "direction": direction,
                    "symbol": "ABC",
                    "lower_date": date,
                    "pre_participation": qualified,
                }
            )
    # Weekly cadence is canonical-sequence adjacency, not seven calendar days.
    for date in pd.to_datetime(["2026-01-02", "2026-01-16"]):
        rows.append(
            {
                "combo": "stocks_b_wmq_all",
                "direction": "LONG",
                "symbol": "XYZ",
                "lower_date": date,
                "pre_participation": True,
            }
        )
    return pd.DataFrame(rows)


def test_episode_start_continuation_end_cadence_direction_and_anchor():
    observations, anchors = construct_preparticipation_episodes(_canonical())
    dwm = observations[observations.combo.eq("stocks_c_dwm_all")]
    assert dwm.episode_id.nunique() == 4  # two episodes for each direction
    assert sorted(dwm.groupby("episode_id").size()) == [1, 1, 2, 2]
    assert len(anchors) == observations.episode_id.nunique()
    assert anchors.episode_anchor.all()
    dwm_anchors = anchors[anchors.combo.eq("stocks_c_dwm_all")]
    assert set(dwm_anchors.groupby(["symbol", "direction"]).size()) == {2}
    wmq = observations[observations.combo.eq("stocks_b_wmq_all")]
    assert wmq.episode_id.nunique() == 1  # adjacent canonical weekly evaluations


def _features():
    return pd.DataFrame(
        {
            "combo": ["stocks_c_dwm_all"] * 3,
            "direction": ["LONG"] * 3,
            "symbol": ["A", "B", "C"],
            "lower_date": pd.to_datetime(["2026-01-01"] * 3),
            "pre_participation": [True] * 3,
            "historical_participation_pass": [True, False, False],
            "upper_has_wyckoff": [True, True, False],
            "lower_sigvol_tier": [1, 2, 2],
            "middle_sigvol_tier": [0, 0, 0],
            "lower_ratio": [0.20, 0.03, 0.075],
            "middle_ratio": [0.0, 0.0, 0.0],
            "immutable_entry_close": [10.0, 20.0, 30.0],
        }
    )


def test_exact_grids_marginal_classification_and_immutable_features():
    source = _features()
    before = source.copy(deep=True)
    candidates = candidate_policies(source)
    assert CANDIDATE_GRIDS == {
        "MODERATE": (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25),
        "STRONG_UPPER_AVAILABLE": (0.02, 0.025, 0.03, 0.04, 0.05),
        "STRONG_UPPER_UNAVAILABLE": (0.025, 0.05, 0.075, 0.10),
    }
    moderate = candidates[
        (candidates.threshold_family == "MODERATE")
        & (candidates.candidate_threshold == 0.15)
    ]
    assert (
        moderate.set_index("symbol").loc["A", "policy_cohort"]
        == "CURRENT_PASS_RETAINED"
    )
    assert candidates.policy_cohort.eq("MARGINAL_NEW_ADMISSIONS").any()
    assert set(candidates.candidate_admission_route) <= {
        "LOWER_ONLY",
        "MIDDLE_ONLY",
        "BOTH",
        "NEITHER",
    }
    assert_frame_equal(source, before)
    assert candidates.immutable_entry_close.notna().all()


def test_observation_robustness_coverage_stale_and_corporate_exclusion():
    frame = pd.DataFrame(
        {
            "combo": ["stocks_c_dwm_all"] * 3,
            "direction": ["LONG"] * 3,
            "horizon_id": ["DWM_1x"] * 3,
            "analysis_unit": ["OBSERVATION"] * 3,
            "historical_cohort": ["CURRENT_PASS"] * 3,
            "theoretically_mature": [True] * 3,
            "terminal_covered": [True, False, True],
            "path_covered": [True, False, True],
            "coverage_status": [
                "MATURE",
                "SYMBOL_HISTORY_STALE_BEFORE_TARGET",
                "MATURE",
            ],
            "corporate_action_flag": [False, False, True],
            "directional_return": [0.1, None, -0.1],
            "mfe": [0.2, None, 0.1],
            "mae": [-0.1, None, -0.2],
        }
    )
    result = summarize_outcomes(frame, ["combo", "historical_cohort"]).iloc[0]
    assert (result.n_total, result.n_mature, result.n_usable) == (3, 3, 2)
    assert result.coverage_rate == 2 / 3 and result.stale_history_count == 1
    assert len(frame[~frame.corporate_action_flag]) == 2


def test_directional_metrics_supported_horizons_and_deterministic_bootstrap():
    assert directional_return(Direction.LONG, 100, 110) > 0
    assert directional_return(Direction.SHORT, 100, 90) > 0
    assert SUPPORTED_HORIZONS["stocks_a_mqy_all"] == frozenset()
    assert "WMQ_3x" not in SUPPORTED_HORIZONS["stocks_b_wmq_all"]
    frame = pd.DataFrame(
        {
            "combo": ["x"] * 4,
            "direction": ["LONG"] * 4,
            "horizon_id": ["h"] * 4,
            "historical_cohort": ["PASS"] * 4,
            "symbol": ["A", "A", "B", "B"],
            "terminal_covered": [True] * 4,
            "directional_return": [0.1, 0.2, -0.1, 0.05],
        }
    )
    groups = ["combo", "direction", "horizon_id", "historical_cohort"]
    assert_frame_equal(
        symbol_cluster_bootstrap(frame, groups, replications=50),
        symbol_cluster_bootstrap(frame, groups, replications=50),
    )


def test_phase4b_has_no_s3_mutation_or_external_provider_path():
    source = inspect.getsource(phase4b)
    assert "put_object" not in source
    assert "delete_object" not in source
    assert "yfinance" not in source
    assert "boto3" not in source
