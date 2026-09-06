import pandas as pd
import pytest

from combos.mtf_scoring_core import evaluate_stocks_options_signal


def scoring_row(
    direction,
    *,
    upper=1.0,
    lower_tier=0,
    lower_ratio=0.0,
    middle_tier=0,
    middle_ratio=0.0
):
    bullish = direction == "long"
    upper_available = not pd.isna(upper)
    return pd.Series(
        {
            "upper_wyckoff_stage": upper if bullish else -upper,
            "upper_exh_abs_pa_prior_bar": 0.0,
            "middle_wyckoff_stage": (
                0.0 if upper_available else (1.0 if bullish else -1.0)
            ),
            "middle_exh_abs_pa_prior_bar": 0.0,
            "middle_sig_vol_current_bar": middle_tier,
            "middle_spy_qqq_vol_ma_ratio": middle_ratio,
            "lower_wyckoff_stage": 0.0,
            "lower_exh_abs_pa_current_bar": 1.0 if bullish else -1.0,
            "lower_sig_vol_current_bar": lower_tier,
            "lower_spy_qqq_vol_ma_ratio": lower_ratio,
            "lower_ma_trend_bullish": 1.0 if bullish else 0.0,
            "lower_ma_trend_bearish": 0.0 if bullish else -1.0,
            "lower_macdv_core_bull": 2.0 if bullish else 0.0,
            "lower_macdv_core_bear": 0.0 if bullish else -2.0,
            "lower_ttm_squeeze_pro": 0.0,
        }
    )


def evaluate(direction, **kwargs):
    row = scoring_row(direction, **kwargs)
    return evaluate_stocks_options_signal(
        row, "lower_exh_abs_pa_current_bar", "lower_sig_vol_current_bar"
    )


@pytest.mark.parametrize("direction", ["long", "short"])
@pytest.mark.parametrize(
    ("upper", "tier", "threshold"),
    [
        (1.0, 2, 0.025),
        (float("nan"), 2, 0.05),
        (1.0, 1, 0.125),
    ],
    ids=["strong-upper-available", "strong-upper-unavailable", "moderate"],
)
def test_production_participation_boundaries_are_strict(
    direction, upper, tier, threshold
):
    for ratio in (threshold - 0.001, threshold):
        signal, long_score, short_score = evaluate(
            direction, upper=upper, lower_tier=tier, lower_ratio=ratio
        )
        assert signal == "none"
        assert (long_score if direction == "long" else short_score) == 4.0

    signal, long_score, short_score = evaluate(
        direction, upper=upper, lower_tier=tier, lower_ratio=threshold + 0.001
    )
    assert signal == direction
    assert (long_score if direction == "long" else short_score) == 5.0


@pytest.mark.parametrize("direction", ["long", "short"])
def test_lower_or_middle_participation_and_sigvol_routing_are_unchanged(direction):
    signal, _, _ = evaluate(
        direction,
        lower_tier=2,
        lower_ratio=0.025,
        middle_tier=2,
        middle_ratio=0.026,
    )
    assert signal == direction

    row = scoring_row(direction, lower_tier=0, lower_ratio=1.0)
    row["lower_sig_vol_prior_bar"] = 2
    signal, _, _ = evaluate_stocks_options_signal(
        row, "lower_exh_abs_pa_current_bar", "lower_sig_vol_prior_bar"
    )
    assert signal == direction


def test_upper_history_route_and_five_of_five_contract_are_unchanged():
    available, available_long, available_short = evaluate(
        "long", upper=1.0, lower_tier=2, lower_ratio=0.026
    )
    unavailable, unavailable_long, unavailable_short = evaluate(
        "long", upper=float("nan"), lower_tier=2, lower_ratio=0.026
    )

    assert (available, available_long, available_short) == ("long", 5.0, 1.0)
    assert (unavailable, unavailable_long, unavailable_short) == ("none", 4.0, 0.0)
