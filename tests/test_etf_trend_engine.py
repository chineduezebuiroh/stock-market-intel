from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combos.mtf_scoring_core import evaluate_stocks_options_signal
from etf import guardrails, trend_engine


def producer_row(**overrides) -> pd.Series:
    values = {
        "symbol": "TEST",
        "wyckoff_stage": 0.0,
        "macdv_guard": 0.0,
        "sig_vol_current_bar": 0.0,
    }
    values.update(overrides)
    return pd.Series(values)


@pytest.mark.parametrize(
    ("tier", "expected"),
    [(0.0, (0.0, 0.0)), (1.0, (1.0, 1.0)), (2.0, (1.0, 1.0)), (np.nan, (0.0, 0.0))],
)
def test_producer_shaped_significant_volume_tiers(tier, expected):
    row = producer_row(sig_vol_current_bar=tier)
    assert "significant_volume" not in row
    assert trend_engine._score_etf_row(row) == expected


def test_directional_components_and_maximum_score():
    bullish = producer_row(wyckoff_stage=2.0, macdv_guard=2.0, sig_vol_current_bar=2.0)
    bearish = producer_row(
        wyckoff_stage=-2.0, macdv_guard=-2.0, sig_vol_current_bar=2.0
    )

    assert trend_engine._score_etf_row(bullish) == (7.0, 1.0)
    assert trend_engine._score_etf_row(bearish) == (1.0, 7.0)


def test_missing_canonical_field_is_missing_data_not_legacy_alias():
    missing = producer_row().drop(labels="sig_vol_current_bar")
    legacy_only = missing.copy()
    legacy_only["significant_volume"] = 2.0

    assert trend_engine._score_etf_row(missing) == (0.0, 0.0)
    assert trend_engine._score_etf_row(legacy_only) == (0.0, 0.0)


@pytest.mark.parametrize("timeframe", ["daily", "weekly"])
def test_compute_scores_uses_same_canonical_contract_for_daily_and_weekly(
    tmp_path, monkeypatch, timeframe
):
    snapshot = pd.DataFrame(
        [
            producer_row(
                symbol="BULL",
                wyckoff_stage=2.0,
                macdv_guard=2.0,
                sig_vol_current_bar=2.0,
            ),
            producer_row(symbol="VOLUME", sig_vol_current_bar=1.0),
        ]
    )
    snapshot.to_parquet(tmp_path / f"snapshot_etf_{timeframe}.parquet")
    monkeypatch.setattr(trend_engine, "DATA", tmp_path)

    scores = trend_engine.compute_etf_trend_scores(timeframe)

    assert scores.loc["BULL"].tolist() == [7.0, 1.0]
    assert scores.loc["VOLUME"].tolist() == [1.0, 1.0]


def admitted_stock_row(**etf_scores) -> pd.Series:
    row = {
        "upper_wyckoff_stage": 1.0,
        "upper_exh_abs_pa_prior_bar": 0.0,
        "middle_wyckoff_stage": 0.0,
        "middle_exh_abs_pa_prior_bar": 0.0,
        "middle_sig_vol_current_bar": 2.0,
        "middle_spy_qqq_vol_ma_ratio": 0.026,
        "lower_wyckoff_stage": 0.0,
        "lower_exh_abs_pa_current_bar": 1.0,
        "lower_sig_vol_current_bar": 0.0,
        "lower_spy_qqq_vol_ma_ratio": 0.0,
        "lower_ma_trend_bullish": 1.0,
        "lower_ma_trend_bearish": 0.0,
        "lower_macdv_core_bull": 2.0,
        "lower_macdv_core_bear": 0.0,
        "lower_ttm_squeeze_pro": 0.0,
    }
    row.update(etf_scores)
    return pd.Series(row)


def evaluate_long_candidate(**etf_scores):
    return evaluate_stocks_options_signal(
        admitted_stock_row(**etf_scores),
        "lower_exh_abs_pa_current_bar",
        "lower_sig_vol_current_bar",
    )[0]


def test_guardrail_threshold_four_changes_confirmation_and_opposition():
    assert evaluate_long_candidate(etf_lower_primary_long_score=3.0) == "watch"
    assert evaluate_long_candidate(etf_lower_primary_long_score=4.0) == "long"
    assert (
        evaluate_long_candidate(
            etf_lower_primary_long_score=4.0,
            etf_lower_primary_short_score=3.0,
        )
        == "long"
    )
    assert (
        evaluate_long_candidate(
            etf_lower_primary_long_score=4.0,
            etf_lower_primary_short_score=4.0,
        )
        == "anti"
    )


def test_attach_etf_trends_preserves_lower_middle_and_alias_contract(
    tmp_path, monkeypatch
):
    pd.DataFrame(
        [
            {
                "symbol": "STOCK",
                "etf_symbol_primary": "PRI",
                "etf_symbol_secondary": "SEC",
            }
        ]
    ).to_csv(tmp_path / "symbol_to_etf_options_eligible.csv", index=False)
    snapshots = {
        "daily": pd.DataFrame(
            [
                producer_row(symbol="PRI", wyckoff_stage=2.0, sig_vol_current_bar=2.0),
                producer_row(symbol="SEC", macdv_guard=-2.0, sig_vol_current_bar=1.0),
            ]
        ),
        "weekly": pd.DataFrame(
            [
                producer_row(symbol="PRI", macdv_guard=2.0, sig_vol_current_bar=1.0),
                producer_row(symbol="SEC", wyckoff_stage=-2.0, sig_vol_current_bar=2.0),
            ]
        ),
    }
    scores = {}
    for timeframe, snapshot in snapshots.items():
        records = []
        for _, row in snapshot.iterrows():
            long_score, short_score = trend_engine._score_etf_row(row)
            records.append((row["symbol"], long_score, short_score))
        scores[timeframe] = pd.DataFrame(
            records,
            columns=["etf_symbol", "etf_long_score", "etf_short_score"],
        ).set_index("etf_symbol")

    monkeypatch.setattr(guardrails, "REF", tmp_path)
    monkeypatch.setattr(guardrails, "load_etf_trend_scores", scores.__getitem__)
    attached = guardrails.attach_etf_trends_for_options_combo(
        pd.DataFrame([{"symbol": "STOCK"}]),
        {"universe": "options_eligible"},
        lower_tf="daily",
        middle_tf="weekly",
    ).iloc[0]

    assert (
        attached.etf_lower_primary_long_score,
        attached.etf_lower_primary_short_score,
    ) == (5.0, 1.0)
    assert (
        attached.etf_lower_secondary_long_score,
        attached.etf_lower_secondary_short_score,
    ) == (1.0, 3.0)
    assert (
        attached.etf_middle_primary_long_score,
        attached.etf_middle_primary_short_score,
    ) == (3.0, 1.0)
    assert (
        attached.etf_middle_secondary_long_score,
        attached.etf_middle_secondary_short_score,
    ) == (1.0, 5.0)
    assert attached.etf_primary_long_score == attached.etf_middle_primary_long_score
    assert attached.etf_primary_short_score == attached.etf_middle_primary_short_score
    assert attached.etf_secondary_long_score == attached.etf_middle_secondary_long_score
    assert (
        attached.etf_secondary_short_score == attached.etf_middle_secondary_short_score
    )
