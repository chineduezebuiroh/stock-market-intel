import json

import pandas as pd
import pytest

from diagnostics.analyze_stock_options_participation import (
    FIVE_COMPONENT_ERA,
    build_directional_opportunities,
)
from diagnostics.analyze_stock_options_phase3b import (
    COMBOS,
    LIMITED_HISTORY_LABEL,
    characterize,
    validate_population,
    write_report,
)


def row(combo, symbol="TEST", date="2026-01-31", ratio=0.051):
    result = {
        "logic_era": FIVE_COMPONENT_ERA,
        "source_s3_key": f"history/{combo}.parquet",
        "artifact_execution_utc": pd.Timestamp("2026-02-01", tz="UTC"),
        "symbol": symbol,
        "lower_date": pd.Timestamp(date),
        "upper_wyckoff_stage": 1.0,
        "pre_participation_long": True,
        "pre_participation_short": True,
        "lower_sig_vol_current_bar": 2,
        "lower_sig_vol_prior_bar": 2,
        "lower_spy_qqq_vol_ma_ratio": ratio,
        "middle_sig_vol_current_bar": 0,
        "middle_spy_qqq_vol_ma_ratio": 0.0,
        "participation_pass": ratio > 0.05,
    }
    return result


def test_all_combo_specs_and_mqy_prior_route_are_used():
    assert set(COMBOS) == {
        "stocks_c_dwm_all",
        "stocks_b_wmq_all",
        "stocks_a_mqy_all",
    }
    source = row("stocks_a_mqy_all")
    source["lower_sig_vol_current_bar"] = 0
    result = build_directional_opportunities(pd.DataFrame([source]), "stocks_a_mqy_all")
    assert set(result.direction) == {"LONG", "SHORT"}
    assert result.lower_participation_pass.all()


def test_characterization_is_combo_directional_and_labels_limited_history(tmp_path):
    frames = {
        combo: pd.DataFrame(
            [row(combo, "PASS", ratio=0.051), row(combo, "BLOCK", ratio=0.05)]
        )
        for combo in COMBOS
    }
    tables = characterize(frames, tmp_path)
    population = tables["population_summary"]
    assert len(population) == 6
    assert set(population.admitted_count) == {1}
    assert set(population.blocker_count) == {1}
    assert not tables["threshold_sensitivity_1d"].empty
    assert not tables["threshold_sensitivity_2d"].empty
    mqy = tables["temporal_summary"].query("combo == 'stocks_a_mqy_all'")
    assert mqy.history_note.eq(LIMITED_HISTORY_LABEL).all()
    write_report(tmp_path, tables)
    assert LIMITED_HISTORY_LABEL in (tmp_path / "phase3b_report.md").read_text()
    assert (tmp_path / "supported_directional_observations.parquet").exists()


def test_growth_safe_guard_and_fail_closed(monkeypatch):
    combo = "stocks_b_wmq_all"
    monkeypatch.setitem(
        __import__(
            "diagnostics.analyze_stock_options_phase3b", fromlist=["GUARDS"]
        ).GUARDS,
        combo,
        (1, 1, "2026-01-03T12:01:33Z", "2026-08-29T12:24:57Z"),
    )
    frame = pd.DataFrame([{"symbol": "A", "lower_date": "2026-01-01"}])
    coverage = {
        "combo": combo,
        "supported_artifact_count": 2,
        "first_supported_five_component_artifact": "2026-01-03T12:01:33Z",
        "last_supported_five_component_artifact": "2026-09-01T00:00:00Z",
        "validation_error_count": 0,
        "scoring_contract": "five_component_with_participation",
        "strict_schema": True,
        "supported_era_contiguous": True,
    }
    validate_population(combo, frame, coverage)
    broken = json.loads(json.dumps(coverage))
    broken["validation_error_count"] = 1
    with pytest.raises(AssertionError, match="validation errors"):
        validate_population(combo, frame, broken)
