import json

import pandas as pd

from diagnostics.analyze_stock_options_participation import (
    COMBO_SPECS,
    FIVE_COMPONENT_ERA,
    classify_modern_scores,
    main,
    provisional_era,
    reconstruct_scores,
)


def scored_row(**overrides):
    row = {
        "symbol": "TEST",
        "lower_date": "2026-08-01",
        "middle_date": "2026-07-01",
        "upper_date": "2026-01-01",
        "upper_wyckoff_stage": 1.0,
        "upper_exh_abs_pa_prior_bar": 0.0,
        "middle_wyckoff_stage": 0.0,
        "middle_exh_abs_pa_prior_bar": 0.0,
        "lower_ma_trend_bullish": 1.0,
        "lower_ma_trend_bearish": 0.0,
        "lower_exh_abs_pa_current_bar": 1.0,
        "lower_exh_abs_pa_prior_bar": 1.0,
        "lower_macdv_core_bull": 2.0,
        "lower_macdv_core_bear": 0.0,
        "lower_ttm_squeeze_pro": 0.0,
        "lower_sig_vol_current_bar": 2.0,
        "lower_sig_vol_prior_bar": 2.0,
        "lower_spy_qqq_vol_ma_ratio": 0.051,
        "middle_sig_vol_current_bar": 0.0,
        "middle_sig_vol_prior_bar": 0.0,
        "middle_spy_qqq_vol_ma_ratio": 0.0,
        "mtf_long_score": 5.0,
        "mtf_short_score": 1.0,
        "signal": "long",
        "signal_side": "long",
        "lower_open": 1.0,
        "lower_high": 2.0,
        "lower_low": 0.5,
        "lower_close": 1.5,
        "lower_volume": 100.0,
        "etf_lower_primary_long_score": float("nan"),
        "etf_lower_primary_short_score": float("nan"),
        "etf_lower_secondary_long_score": float("nan"),
        "etf_lower_secondary_short_score": float("nan"),
        "etf_primary_long_score": float("nan"),
        "etf_primary_short_score": float("nan"),
        "etf_secondary_long_score": float("nan"),
        "etf_secondary_short_score": float("nan"),
    }
    row.update(overrides)
    return row


def reconstruct(combo, **overrides):
    frame = pd.DataFrame([scored_row(**overrides)])
    return reconstruct_scores(frame, COMBO_SPECS[combo])[0].iloc[0]


def test_current_bar_routing_for_dwm_and_wmq_is_unchanged():
    for combo in ("stocks_c_dwm_all", "stocks_b_wmq_all"):
        row = reconstruct(
            combo,
            lower_exh_abs_pa_current_bar=0.0,
            lower_exh_abs_pa_prior_bar=1.0,
            lower_sig_vol_current_bar=0.0,
            lower_sig_vol_prior_bar=2.0,
        )
        assert not row.price_action_long_pass
        assert not row.lower_route_pass


def test_mqy_uses_prior_bar_routing_and_wrong_current_route_mismatches():
    mqy = reconstruct(
        "stocks_a_mqy_all",
        lower_exh_abs_pa_current_bar=0.0,
        lower_exh_abs_pa_prior_bar=1.0,
        lower_sig_vol_current_bar=0.0,
        lower_sig_vol_prior_bar=2.0,
    )
    wrong = reconstruct(
        "stocks_c_dwm_all",
        lower_exh_abs_pa_current_bar=0.0,
        lower_exh_abs_pa_prior_bar=1.0,
        lower_sig_vol_current_bar=0.0,
        lower_sig_vol_prior_bar=2.0,
    )
    assert mqy.price_action_long_pass and mqy.lower_route_pass
    assert mqy.long_score_match
    assert not wrong.long_score_match


def test_strict_thresholds_and_lower_or_middle_are_exact():
    at_five = reconstruct("stocks_b_wmq_all", lower_spy_qqq_vol_ma_ratio=0.05)
    no_upper_at_ten = reconstruct(
        "stocks_b_wmq_all",
        upper_wyckoff_stage=float("nan"),
        lower_spy_qqq_vol_ma_ratio=0.10,
        mtf_long_score=4.0,
        mtf_short_score=0.0,
    )
    at_twenty_five = reconstruct(
        "stocks_b_wmq_all",
        lower_sig_vol_current_bar=1.0,
        lower_spy_qqq_vol_ma_ratio=0.25,
        middle_sig_vol_current_bar=2.0,
        middle_spy_qqq_vol_ma_ratio=0.051,
    )
    assert not at_five.lower_route_pass
    assert not no_upper_at_ten.lower_route_pass
    assert not at_twenty_five.lower_route_pass
    assert at_twenty_five.middle_route_pass and at_twenty_five.participation_pass


def test_score_mismatch_is_quarantined_and_schema_is_combo_specific():
    frame = pd.DataFrame([scored_row(mtf_long_score=3.0)])
    reconstructed, _ = reconstruct_scores(frame, COMBO_SPECS["stocks_b_wmq_all"])
    assert classify_modern_scores(reconstructed) == "MODERN_QUARANTINED_SCORE_MISMATCH"

    fields = set(frame.columns) - {"lower_exh_abs_pa_prior_bar"}
    era, missing = provisional_era(fields, COMBO_SPECS["stocks_a_mqy_all"])
    assert era == "UNKNOWN_OR_MIXED"
    assert "lower_exh_abs_pa_prior_bar" in missing


def test_validation_canonicalizes_latest_and_marks_incomplete(tmp_path):
    history = tmp_path / "history"
    output = tmp_path / "output"
    history.mkdir()
    rows = [scored_row(symbol=f"S{i}") for i in range(4)]
    pd.DataFrame(rows).to_parquet(
        history / "combo_stocks_b_wmq_all_asof=2026-08-01T01-00-00.parquet"
    )
    later = pd.DataFrame(rows)
    later["lower_close"] = 2.0
    later.to_parquet(
        history / "combo_stocks_b_wmq_all_asof=2026-08-01T02-00-00.parquet"
    )
    pd.DataFrame([scored_row(symbol="OUTLIER")]).to_parquet(
        history / "combo_stocks_b_wmq_all_asof=2026-08-01T03-00-00.parquet"
    )

    assert (
        main(
            [
                "--combo",
                "stocks_b_wmq_all",
                "--output-dir",
                str(output),
                "--local-history-dir",
                str(history),
                "--phase",
                "validate",
                "--strict-schema",
            ]
        )
        == 0
    )
    coverage = json.loads((output / "coverage_summary.json").read_text())
    canonical = pd.read_parquet(output / "supported_observations.parquet")
    assert coverage["raw_artifact_executions"] == 3
    assert coverage["suspiciously_incomplete_artifacts"] == 1
    assert coverage["canonical_lower_market_dates"] == 1
    assert len(canonical) == 4
    assert set(canonical["artifact_execution_utc"]) == {
        pd.Timestamp("2026-08-01T02:00:00Z")
    }
    assert coverage["readiness_decision"].startswith("A —")
    for name in (
        "artifact_inventory.csv",
        "schema_era_summary.csv",
        "canonicalization_summary.json",
        "canonicalization_summary.csv",
        "score_reconstruction_summary.csv",
        "etf_coverage_summary.csv",
        "validation_errors.csv",
        "validation_report.md",
    ):
        assert (output / name).exists()


def test_exact_reconstruction_classifies_five_component_candidate():
    frame = pd.DataFrame([scored_row()])
    reconstructed, _ = reconstruct_scores(frame, COMBO_SPECS["stocks_c_dwm_all"])
    assert classify_modern_scores(reconstructed) == FIVE_COMPONENT_ERA


def test_incompatible_post_boundary_schema_fails_closed(tmp_path):
    history = tmp_path / "history"
    output = tmp_path / "output"
    history.mkdir()
    rows = [scored_row(symbol=f"S{i}") for i in range(2)]
    pd.DataFrame(rows).to_parquet(
        history / "combo_stocks_a_mqy_all_asof=2026-08-01T01-00-00.parquet"
    )
    incompatible = pd.DataFrame(rows).drop(columns="lower_sig_vol_prior_bar")
    incompatible.to_parquet(
        history / "combo_stocks_a_mqy_all_asof=2026-08-02T01-00-00.parquet"
    )

    assert (
        main(
            [
                "--combo",
                "stocks_a_mqy_all",
                "--output-dir",
                str(output),
                "--local-history-dir",
                str(history),
                "--phase",
                "validate",
                "--strict-schema",
            ]
        )
        == 2
    )
    errors = pd.read_csv(output / "validation_errors.csv")
    assert errors["error"].str.contains("post-boundary artifact").any()
