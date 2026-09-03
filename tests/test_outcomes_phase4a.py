from dataclasses import replace
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from diagnostics.outcomes.horizons import resolve_target_date
from diagnostics.outcomes.metrics import (
    PriceBar,
    calculate_metrics,
    directional_return,
    excursions,
)
from diagnostics.outcomes.price_source import RollingDailyPriceSource
from diagnostics.outcomes.regimes import assign_engine_regime
from diagnostics.outcomes.specs import (
    Direction,
    EntryState,
    HORIZON_SPECS,
    MaturityStatus,
    PolicyMode,
    evaluate_policy,
    target_date,
)


def entry() -> EntryState:
    return EntryState(
        "stocks_c_dwm_all",
        Direction.LONG,
        "XYZ",
        date(2026, 1, 2),
        datetime(2026, 1, 2, 23, tzinfo=timezone.utc),
        "immutable/key.parquet",
        "dwm_modern_five_component_v1",
        True,
        False,
        "NONE",
        1,
        2,
        0.2,
        0.3,
        True,
        4,
        1,
        100,
    )


def test_all_combo_horizon_specs_and_primary_designations():
    expected = {
        "stocks_c_dwm_all": [30, 60, 90],
        "stocks_b_wmq_all": [90, 180, 270],
        "stocks_a_mqy_all": [365, 730, 1095],
    }
    assert {
        key: [h.calendar_days for h in value.horizons]
        for key, value in HORIZON_SPECS.items()
    } == expected
    for spec in HORIZON_SPECS.values():
        assert spec.horizons[0].primary_directions == {Direction.SHORT}
        assert all(h.primary_directions == {Direction.LONG} for h in spec.horizons[1:])


def test_weekend_and_holiday_alignment_is_forward_only_and_bounded():
    target = date(2026, 7, 4)
    result = resolve_target_date(
        target, [date(2026, 7, 2), date(2026, 7, 6)], data_asof=date(2026, 7, 6)
    )
    assert result.resolved_date == date(2026, 7, 6)
    assert result.status is MaturityStatus.MATURE
    assert result.resolved_date >= target


def test_maturity_missing_and_unresolvable_classification():
    target = date(2026, 8, 1)
    assert (
        resolve_target_date(target, [], data_asof=date(2026, 9, 1)).status
        is MaturityStatus.MISSING_PRICE_HISTORY
    )
    assert (
        resolve_target_date(
            target, [date(2026, 7, 31)], data_asof=date(2026, 7, 31)
        ).status
        is MaturityStatus.IMMATURE
    )
    assert (
        resolve_target_date(
            target, [date(2026, 7, 31)], data_asof=date(2026, 8, 9)
        ).status
        is MaturityStatus.UNRESOLVABLE_TARGET_DATE
    )


def test_long_and_short_returns_and_excursions_preserve_signs():
    bars = [
        PriceBar(date(2026, 1, 5), 110, 95, 105),
        PriceBar(date(2026, 1, 6), 108, 80, 90),
    ]
    assert directional_return(Direction.LONG, 100, 90) == pytest.approx(-0.1)
    assert directional_return(Direction.SHORT, 100, 90) == pytest.approx(100 / 90 - 1)
    assert excursions(Direction.LONG, 100, bars) == pytest.approx((0.1, -0.2))
    assert excursions(Direction.SHORT, 100, bars) == pytest.approx(
        (0.25, 100 / 110 - 1)
    )
    result = calculate_metrics(Direction.LONG, date(2026, 1, 2), 100, bars[-1], bars)
    assert (result.elapsed_calendar_days, result.elapsed_trading_sessions) == (4, 2)


def test_price_source_missing_and_normalization_are_read_only():
    empty = RollingDailyPriceSource(lambda _: pd.DataFrame()).load("gone")
    assert empty.frame.empty and empty.source_key == "bars/stocks_daily/GONE.parquet"
    original = pd.DataFrame(
        {"high": [2], "low": [1], "close": [1.5]},
        index=[pd.Timestamp("2026-01-02", tz="UTC")],
    )
    loaded = RollingDailyPriceSource(lambda _: original).load("xyz")
    assert loaded.frame.index.tz is None
    assert original.index.tz is not None


def test_immutable_entry_and_policy_modes_are_separate():
    immutable = entry()
    historical = evaluate_policy(immutable, PolicyMode.HISTORICAL_PRODUCTION)
    counterfactual = evaluate_policy(
        immutable,
        PolicyMode.COUNTERFACTUAL_PARTICIPATION,
        participation_pass=True,
        participation_route="LOWER",
        parameters={"moderate": 0.2},
    )
    assert not historical.participation_pass and counterfactual.participation_pass
    assert counterfactual.entry_state == immutable
    assert (
        immutable.historical_participation_pass is False
        and immutable.entry_close == 100
    )
    with pytest.raises(ValueError):
        evaluate_policy(
            immutable, PolicyMode.HISTORICAL_PRODUCTION, participation_pass=True
        )


def test_engine_regime_assignment_is_deterministic_and_fails_closed():
    ts = datetime(2026, 1, 5, tzinfo=timezone.utc)
    assert (
        assign_engine_regime("stocks_c_dwm_all", ts) == "dwm_modern_five_component_v1"
    )
    assert (
        assign_engine_regime("stocks_b_wmq_all", ts) == "wmq_modern_five_component_v1"
    )
    assert (
        assign_engine_regime("stocks_a_mqy_all", ts) == "mqy_modern_five_component_v1"
    )
    assert (
        assign_engine_regime(
            "stocks_c_dwm_all", ts, observed_logic_era="LEGACY_COMBINED_MACDV"
        )
        == "legacy_combined_macdv"
    )
    with pytest.raises(ValueError):
        assign_engine_regime(
            "stocks_a_mqy_all", datetime(2026, 1, 1, tzinfo=timezone.utc)
        )


def test_mqy_long_horizons_are_immature_without_fabrication():
    spec = HORIZON_SPECS["stocks_a_mqy_all"]
    for horizon in spec.horizons[1:]:
        target = target_date(date(2026, 1, 30), horizon)
        result = resolve_target_date(
            target, [date(2026, 9, 1)], data_asof=date(2026, 9, 1)
        )
        assert result.status is MaturityStatus.IMMATURE
        assert result.resolved_date is None


def test_outcome_package_has_no_production_storage_or_mutation_imports():
    import diagnostics.outcomes.price_source as module

    source = open(module.__file__).read()
    assert "core.storage" not in source
    assert "boto3" not in source
    assert "save_" not in source and "delete_" not in source
