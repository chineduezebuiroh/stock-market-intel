from datetime import date

import numpy as np
import pandas as pd
import pytest

from etl.eod_reconciliation import (
    TerminalObservation,
    canonical_terminal_session_close,
    observation_from_native_frame,
    repair_terminal_close,
)

SESSION = date(2026, 9, 23)


def obs(interval, close=100.0, session=SESSION, **kwargs):
    return TerminalObservation(
        interval, "2026-09-01", session,
        kwargs.pop("session_evidence", "test-explicit-session"), close, **kwargs
    )


def frame(close=np.nan, *, low=99.0, high=101.0, bad=None):
    result = pd.DataFrame(
        {"open": [99.5], "high": [high], "low": [low], "close": [close],
         "adj_close": [np.nan], "volume": [1000.0]},
        index=[pd.Timestamp("2026-09-23")],
    )
    if bad:
        result.loc[result.index[-1], bad] = np.nan
    return result


@pytest.mark.parametrize(
    "target,sources",
    [("1d", ("1wk", "1mo")), ("1wk", ("1d", "1mo")), ("1mo", ("1d", "1wk"))],
)
def test_each_malformed_target_can_be_repaired_by_other_two(target, sources):
    result = canonical_terminal_session_close([obs(source) for source in sources],
                                              target_session=SESSION, price_tolerance=.005)
    repaired, provenance = repair_terminal_close(frame(), result)
    assert repaired.iloc[-1].close == 100
    assert provenance["data_quality_status"] == "repaired"
    assert provenance["raw_observed_close"] is None
    assert set(provenance["corroborating_sources"]) == set(sources)


def test_all_valid_agree_is_canonical_but_observed_rows_need_no_repair():
    result = canonical_terminal_session_close([obs("1d"), obs("1wk"), obs("1mo")],
                                              target_session=SESSION, price_tolerance=.005)
    assert result.status == "canonical"
    assert result.canonical_close == 100
    assert result.evidence_tier == "tier_1_independent_consensus"


def test_two_agree_third_contradicts_vetoes_repair():
    result = canonical_terminal_session_close(
        [obs("1d"), obs("1wk"), obs("1mo", 101)], target_session=SESSION, price_tolerance=.005)
    assert result.status == "conflict"
    assert result.canonical_close is None
    assert result.conflicting_sources == ("1mo",)


def test_only_one_source_fails_closed():
    assert canonical_terminal_session_close([obs("1wk")], target_session=SESSION,
                                            price_tolerance=.005).status == "excluded"


@pytest.mark.parametrize("interval", ["1wk", "1mo"])
def test_one_price_bound_coarse_source_is_symmetric_tier_2(interval):
    source = obs(interval, session_evidence="yfinance_interval_merge_lastTrade")
    result = canonical_terminal_session_close(
        [source], target_session=SESSION, price_tolerance=.005
    )
    assert result.status == "canonical"
    assert result.evidence_tier == "tier_2_provenance_bound_single_source"
    assert result.canonical_close == 100


def test_equal_close_with_session_mismatch_fails_closed():
    result = canonical_terminal_session_close(
        [obs("1wk"), obs("1mo", session=date(2026, 9, 22))],
        target_session=SESSION, price_tolerance=.005)
    assert result.status == "excluded"


@pytest.mark.parametrize("low,high,bad", [(101, 102, None), (99, 101, "open"), (99, 101, "volume")])
def test_target_bounds_and_ohlcv_are_mandatory(low, high, bad):
    result = canonical_terminal_session_close([obs("1wk"), obs("1mo")],
                                              target_session=SESSION, price_tolerance=.005)
    repaired, provenance = repair_terminal_close(frame(low=low, high=high, bad=bad), result)
    assert pd.isna(repaired.iloc[-1].close)
    assert provenance["data_quality_status"] == "excluded"


def test_target_open_must_also_be_inside_native_range():
    target = frame()
    target.loc[target.index[-1], "open"] = 102
    result = canonical_terminal_session_close([obs("1wk"), obs("1mo")],
                                              target_session=SESSION, price_tolerance=.005)
    repaired, provenance = repair_terminal_close(target, result)
    assert pd.isna(repaired.iloc[-1].close)
    assert provenance["data_quality_status"] == "excluded"


def test_adjusted_or_non_regular_semantics_are_ineligible():
    adjusted = obs("1wk", adjustment_mode="adjusted")
    extended = obs("1mo", session_semantics="extended")
    result = canonical_terminal_session_close([adjusted, extended], target_session=SESSION,
                                              price_tolerance=.005)
    assert result.status == "excluded"


def test_same_bucket_label_does_not_make_stale_row_eligible():
    stale = obs("1wk", session=date(2026, 9, 21))
    current = obs("1mo")
    assert stale.interval_label == current.interval_label
    result = canonical_terminal_session_close([stale, current], target_session=SESSION,
                                              price_tolerance=.005)
    assert result.status == "excluded"


def test_tolerance_boundary_is_inclusive_and_above_is_conflict():
    at = canonical_terminal_session_close([obs("1wk", 100), obs("1mo", 100.005)],
                                          target_session=SESSION, price_tolerance=.005)
    above = canonical_terminal_session_close([obs("1wk", 100), obs("1mo", 100.00501)],
                                             target_session=SESSION, price_tolerance=.005)
    assert at.status == "canonical"
    assert above.status == "conflict"


def test_tier_1_prefers_finer_price_precision_then_finer_interval():
    weekly = obs("1wk", 100.00, price_tolerance=.005)
    monthly = obs("1mo", 100.01, price_tolerance=.05)
    precise = canonical_terminal_session_close(
        [weekly, monthly], target_session=SESSION, price_tolerance=.05
    )
    assert precise.canonical_close == 100.00

    weekly = obs("1wk", 100.00, price_tolerance=.05)
    monthly = obs("1mo", 100.01, price_tolerance=.05)
    finer_interval = canonical_terminal_session_close(
        [monthly, weekly], target_session=SESSION, price_tolerance=.05
    )
    assert finer_interval.canonical_close == 100.00
    assert "selected 1wk" in finer_interval.reason


def test_revision_ambiguity_fails_closed():
    result = canonical_terminal_session_close(
        [obs("1wk"), obs("1mo", revision_ambiguous=True)],
        target_session=SESSION, price_tolerance=.005)
    assert result.status == "excluded"


@pytest.mark.parametrize("interval", ["3mo", "1y", "4h"])
def test_derived_q_y_and_intraday_4h_are_never_eligible(interval):
    result = canonical_terminal_session_close([obs("1d"), obs(interval)],
                                              target_session=SESSION, price_tolerance=.005)
    assert result.status == "excluded"


def test_live_pattern_uses_verified_interval_specific_last_trade():
    peers = []
    for interval in ("1wk", "1mo"):
        peer = frame(100)
        peer.index = [pd.Timestamp("2026-09-21" if interval == "1wk" else "2026-09-01")]
        peer.attrs["eod_provenance"] = {
            "source_interval": interval, "adjustment_mode": "raw_unadjusted",
            "session_semantics": "regular",
            "provider_metadata": {"lastTrade": {"Price": 100, "Time": pd.Timestamp("2026-09-23")}},
        }
        peers.append(observation_from_native_frame(peer))
    result = canonical_terminal_session_close(peers, target_session=SESSION, price_tolerance=.005)
    repaired, provenance = repair_terminal_close(frame(), result)
    assert repaired.iloc[-1].close == 100
    assert provenance["close_source"] == "1mo+1wk"


def test_last_trade_price_must_bind_to_returned_aggregate():
    peer = frame(100)
    peer.attrs["eod_provenance"] = {
        "source_interval": "1wk", "adjustment_mode": "raw_unadjusted",
        "session_semantics": "regular",
        "provider_metadata": {"lastTrade": {"Price": 99, "Time": pd.Timestamp("2026-09-23")}},
    }
    assert observation_from_native_frame(peer).terminal_market_session_date is None


def test_weekly_last_trade_does_not_make_null_weekly_close_an_eligible_source():
    peer = frame(np.nan)
    peer.attrs["eod_provenance"] = {
        "source_interval": "1wk", "adjustment_mode": "raw_unadjusted",
        "session_semantics": "regular", "provider_metadata": {
            "priceHint": 2,
            "lastTrade": {"Price": 100, "Time": pd.Timestamp("2026-09-23")},
        },
    }
    observation = observation_from_native_frame(peer)
    assert observation.close is None
    assert observation.terminal_market_session_date is None
    assert observation.session_evidence is None


def test_close_only_repair_leaves_adjusted_close_and_other_fields_untouched():
    original = frame()
    result = canonical_terminal_session_close([obs("1wk"), obs("1mo")],
                                              target_session=SESSION, price_tolerance=.005)
    repaired, _ = repair_terminal_close(original, result)
    pd.testing.assert_series_equal(repaired.iloc[-1].drop("close"), original.iloc[-1].drop("close"))
