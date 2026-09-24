import numpy as np
import pandas as pd
import pytest

from etl import sources
from etl.eod_quality import (
    StockEODDataQualityError,
    merge_stock_eod_window,
    require_valid_terminal_stock_eod,
    valid_stock_eod_rows,
)


def bars(close=(99.0, 100.0), *, start="2026-09-18"):
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "open": [value - 1 for value in close],
            "high": [value + 1 for value in close],
            "low": [value - 2 for value in close],
            "close": close,
            "adj_close": close,
            "volume": [1_000] * len(close),
        },
        index=index,
    )


def test_missing_provider_close_is_invalid():
    frame = bars().drop(columns="close")
    assert not valid_stock_eod_rows(frame).any()
    with pytest.raises(StockEODDataQualityError, match="close"):
        require_valid_terminal_stock_eod(frame, context="test")


def test_load_eod_synthesizes_unrecognized_close_as_missing(monkeypatch):
    provider = bars().rename(columns={"close": "Settlement"})

    class FakeTicker:
        def history(self, **kwargs):
            return provider

    monkeypatch.setattr(sources.yf, "Ticker", lambda symbol: FakeTicker())
    loaded = sources.load_eod("CRWD", timeframe="daily", window_bars=260)

    assert loaded["close"].isna().all()
    assert loaded[["open", "high", "low", "volume"]].notna().all().all()
    assert not valid_stock_eod_rows(loaded).any()


def test_nan_terminal_close_is_invalid_and_cannot_proceed():
    frame = bars()
    frame.iloc[-1, frame.columns.get_loc("close")] = np.nan
    with pytest.raises(StockEODDataQualityError, match="close"):
        merge_stock_eod_window(frame, pd.DataFrame(), 260)


def test_malformed_duplicate_retains_entire_valid_existing_row():
    existing = bars()
    new = existing.iloc[[-1]].copy()
    new.loc[:, ["open", "high", "low", "volume"]] = [500, 501, 499, 999_999]
    new.loc[:, "close"] = np.nan

    result = merge_stock_eod_window(new, existing, 260)

    pd.testing.assert_series_equal(result.frame.iloc[-1], existing.iloc[-1])
    assert result.retained_existing_timestamps == (existing.index[-1],)


def test_valid_duplicate_replaces_existing_unchanged_as_a_whole_row():
    existing = bars()
    new = existing.iloc[[-1]].copy()
    new.loc[:, ["open", "high", "low", "close", "adj_close", "volume"]] = [
        101,
        104,
        100,
        103,
        102.5,
        2_000,
    ]
    result = merge_stock_eod_window(new, existing, 260)
    pd.testing.assert_series_equal(result.frame.iloc[-1], new.iloc[-1])
    assert not result.dropped_malformed_timestamps


@pytest.mark.parametrize("date", ["2026-09-21"])
def test_monday_partial_daily_weekly_monthly_rows_are_valid(date):
    frames = [bars((249.350006,), start=date) for _ in range(3)]
    for frame in frames:
        require_valid_terminal_stock_eod(frame, context="partial interval")


def test_same_session_other_timeframe_close_does_not_repair_invalid_row():
    daily = bars((np.nan,), start="2026-09-21")
    monthly = bars((249.350006,), start="2026-09-21")
    require_valid_terminal_stock_eod(monthly, context="monthly")
    with pytest.raises(StockEODDataQualityError):
        merge_stock_eod_window(daily, pd.DataFrame(), 260)
