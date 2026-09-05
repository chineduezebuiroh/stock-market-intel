import inspect
import time

import numpy as np
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


def _reference_bootstrap(frame, groups, reps):
    """Original Phase 4C implementation retained as an equivalence oracle."""
    rng = np.random.default_rng(phase4c.BOOTSTRAP_SEED)
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        usable = part[part.terminal_covered.astype(bool)]
        symbols = usable.symbol.dropna().unique()
        draws = []
        for _ in range(reps if len(symbols) else 0):
            sampled_symbols = rng.choice(symbols, len(symbols), replace=True)
            sampled = pd.concat(
                [usable[usable.symbol.eq(symbol)] for symbol in sampled_symbols]
            )
            draws.append(
                [
                    sampled.directional_return.mean(),
                    sampled.directional_return.median(),
                    sampled.directional_return.gt(0).mean(),
                    sampled.mfe.median(),
                    sampled.mae.median(),
                ]
            )
        row = dict(zip(groups, keys if isinstance(keys, tuple) else (keys,))) | {
            "bootstrap_seed": phase4c.BOOTSTRAP_SEED,
            "bootstrap_replications": reps,
            "cluster": "symbol",
            "n_symbols": len(symbols),
        }
        for index, metric in enumerate(phase4c.METRICS):
            values = np.asarray(draws)[:, index] if draws else np.array([])
            row[f"{metric}_ci_low"] = (
                np.quantile(values, 0.025) if len(values) else np.nan
            )
            row[f"{metric}_ci_high"] = (
                np.quantile(values, 0.975) if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_fixture(groups=2, symbols=5, rows_per_symbol=3):
    rows = []
    for group in range(groups):
        for symbol in range(symbols):
            for observation in range(rows_per_symbol + symbol % 2):
                value = (group + 1) * (symbol - 2) / 100 + observation / 1000
                rows.append(
                    {
                        "group": f"g{group}",
                        "symbol": f"s{symbol}",
                        "terminal_covered": observation != 3,
                        "directional_return": value,
                        "mfe": value + 0.1,
                        "mae": value - 0.1,
                    }
                )
    # Exercise the exact pandas missing-value reduction semantics too.
    rows[1]["directional_return"] = np.nan
    rows[2]["mfe"] = np.nan
    rows[3]["mae"] = np.nan
    return pd.DataFrame(rows)


def test_optimized_bootstrap_matches_reference_sample_and_all_intervals():
    frame = _bootstrap_fixture()
    expected = _reference_bootstrap(frame, ["group"], reps=137)
    actual = phase4c.bootstrap_intervals(frame, ["group"], reps=137)
    assert_frame_equal(actual, expected, check_exact=False, atol=1e-14, rtol=1e-14)
    assert list(actual.filter(like="_ci_").columns) == [
        f"{metric}_{bound}"
        for metric in phase4c.METRICS
        for bound in ("ci_low", "ci_high")
    ]


def test_bootstrap_repeated_seed_matches_reference():
    frame = _bootstrap_fixture(groups=3)
    first = phase4c.bootstrap_intervals(frame, ["group"], reps=51)
    second = phase4c.bootstrap_intervals(frame, ["group"], reps=51)
    reference = _reference_bootstrap(frame, ["group"], reps=51)
    assert_frame_equal(first, second)
    assert_frame_equal(first, reference, check_exact=False, atol=1e-14, rtol=1e-14)


def test_bootstrap_nontrivial_fixture_timing_smoke():
    """Exercise realistic nested work without a brittle wall-clock assertion."""
    frame = _bootstrap_fixture(groups=8, symbols=20, rows_per_symbol=8)
    started = time.perf_counter()
    result = phase4c.bootstrap_intervals(frame, ["group"], reps=100)
    elapsed = time.perf_counter() - started
    assert len(result) == 8
    assert result.bootstrap_replications.eq(100).all()
    print(f"optimized bootstrap smoke: {elapsed:.3f}s")


def test_bootstrap_progress_is_bounded(capsys):
    phase4c.bootstrap_intervals(
        _bootstrap_fixture(groups=12), ["group"], reps=2, progress=True
    )
    messages = capsys.readouterr().out.strip().splitlines()
    assert messages[-1] == "[BOOTSTRAP] completed cohort 12/12"
    assert len(messages) <= 12


def test_atomic_writer_replaces_partial_output(tmp_path):
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("partial")
    expected = pd.DataFrame({"value": [1, 2]})
    phase4c._write_csv(expected, artifact)
    assert_frame_equal(pd.read_csv(artifact), expected)
    assert list(tmp_path.iterdir()) == [artifact]
