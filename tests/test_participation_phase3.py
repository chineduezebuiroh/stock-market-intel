import pandas as pd
import pytest

from diagnostics.analyze_stock_options_participation import (
    FIVE_COMPONENT_ERA,
    build_directional_opportunities,
    construct_episodes,
    scenario_pass,
    transition_table,
)


def canonical_row(**overrides):
    row = {
        "logic_era": FIVE_COMPONENT_ERA,
        "symbol": "TEST",
        "lower_date": pd.Timestamp("2026-01-02"),
        "upper_wyckoff_stage": 1.0,
        "pre_participation_long": True,
        "pre_participation_short": False,
        "lower_sig_vol_current_bar": 2,
        "lower_spy_qqq_vol_ma_ratio": 0.051,
        "middle_sig_vol_current_bar": 0,
        "middle_spy_qqq_vol_ma_ratio": 1.0,
        "participation_pass": True,
    }
    row.update(overrides)
    return row


def opportunities(**overrides):
    return build_directional_opportunities(pd.DataFrame([canonical_row(**overrides)]))


def test_current_rule_strict_threshold_and_tier_branches():
    assert (
        opportunities()
        .query("direction == 'LONG'")["overall_participation_pass"]
        .item()
    )
    assert (
        not opportunities(lower_spy_qqq_vol_ma_ratio=0.05, participation_pass=False)
        .query("direction == 'LONG'")["overall_participation_pass"]
        .item()
    )
    assert (
        opportunities(lower_sig_vol_current_bar=1, lower_spy_qqq_vol_ma_ratio=0.251)
        .query("direction == 'LONG'")["lower_participation_pass"]
        .item()
    )
    assert (
        not opportunities(
            lower_sig_vol_current_bar=1,
            lower_spy_qqq_vol_ma_ratio=0.25,
            participation_pass=False,
        )
        .query("direction == 'LONG'")["lower_participation_pass"]
        .item()
    )
    assert (
        opportunities(upper_wyckoff_stage=None, lower_spy_qqq_vol_ma_ratio=0.101)
        .query("direction == 'LONG'")["lower_participation_pass"]
        .item()
    )
    assert (
        not opportunities(
            upper_wyckoff_stage=None,
            lower_spy_qqq_vol_ma_ratio=0.10,
            participation_pass=False,
        )
        .query("direction == 'LONG'")["lower_participation_pass"]
        .item()
    )


def test_middle_rescue_and_no_eligible_route():
    rescued = opportunities(
        lower_spy_qqq_vol_ma_ratio=0.01,
        middle_sig_vol_current_bar=2,
        middle_spy_qqq_vol_ma_ratio=0.06,
    )
    assert rescued.query("direction == 'LONG'")["route_class"].item() == "MIDDLE_ONLY"
    none = opportunities(
        lower_sig_vol_current_bar=0,
        middle_sig_vol_current_bar=0,
        participation_pass=False,
    )
    long = none.query("direction == 'LONG'").iloc[0]
    assert long["eligibility_state"] == "NO_ELIGIBLE_SIGVOL_ROUTE"
    assert pd.isna(long["best_threshold_normalized_ratio"])


def test_distance_math_and_sensitivity_have_no_double_counting():
    frame = opportunities(
        lower_spy_qqq_vol_ma_ratio=0.04,
        middle_sig_vol_current_bar=2,
        middle_spy_qqq_vol_ma_ratio=0.03,
        participation_pass=False,
    )
    long = frame.query("direction == 'LONG'").iloc[0]
    assert long["best_threshold_normalized_ratio"] == pytest.approx(0.8)
    assert long["best_absolute_margin"] == pytest.approx(-0.01)
    assert (
        scenario_pass(frame, strong_available=0.025).sum() == 2
    )  # one per direction, never one per route
    assert scenario_pass(frame, strong_available=0.05).sum() == 0
    assert scenario_pass(frame, strong_available=0.025, moderate=0.05).sum() == 2


def test_episode_construction_and_transition_classification():
    rows = []
    for day, passed, pre in [
        (2, False, True),
        (3, True, True),
        (6, False, False),
        (7, True, True),
    ]:
        rows.append(
            canonical_row(
                lower_date=pd.Timestamp(f"2026-01-{day:02d}"),
                lower_spy_qqq_vol_ma_ratio=0.06 if passed else 0.01,
                participation_pass=passed,
                pre_participation_long=pre,
            )
        )
    directional = build_directional_opportunities(pd.DataFrame(rows))
    episodes, _ = construct_episodes(directional)
    long = episodes.query("direction == 'LONG'")
    assert len(long) == 2
    assert long.iloc[0]["mixed_pass_fail"]
    transitions = transition_table(directional)
    assert "BLOCK -> PASS" in set(transitions["transition"])


def test_unsupported_era_and_duplicate_exclusion():
    bad = pd.DataFrame([canonical_row(logic_era="MODERN_PRE_PARTICIPATION_SCORE")])
    try:
        build_directional_opportunities(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("unsupported era was accepted")
    duplicate = pd.DataFrame([canonical_row(), canonical_row()])
    try:
        build_directional_opportunities(duplicate)
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate was accepted")
