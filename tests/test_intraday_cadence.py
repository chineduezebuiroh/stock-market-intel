from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from core.intraday_cadence import (
    MAX_SCHEDULE_DELAY_MINUTES,
    RETRY_MINUTES,
    is_futures_4h_opportunity,
    is_futures_4h_target_hour,
    is_retry_opportunity,
    is_stocks_4h_opportunity,
    is_stocks_4h_target_hour,
)

NY = ZoneInfo("America/New_York")


def ny(year, month, day, hour, minute=1):
    return datetime(year, month, day, hour, minute, tzinfo=NY)


@pytest.mark.parametrize("hour", [1, 5, 9, 13, 17, 21])
@pytest.mark.parametrize("day", [5, 6, 7, 8])  # Mon-Thu, EST
def test_futures_weekday_target_hours_est(day, hour):
    assert is_futures_4h_target_hour(ny(2026, 1, day, hour))


@pytest.mark.parametrize("hour", [0, 2, 8, 10, 16, 18, 22])
def test_futures_non_target_hours_rejected(hour):
    assert not is_futures_4h_target_hour(ny(2026, 7, 6, hour))  # Monday, EDT


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (ny(2026, 7, 5, 21), True),  # Sunday EDT
        (ny(2026, 7, 5, 17), False),
        (ny(2026, 7, 10, 13), True),  # Friday
        (ny(2026, 7, 10, 17), False),
        (ny(2026, 7, 11, 9), False),  # Saturday
    ],
)
def test_futures_session_day_boundaries(value, expected):
    assert is_futures_4h_target_hour(value) is expected


@pytest.mark.parametrize("hour", [9, 13, 17])
@pytest.mark.parametrize("day", [6, 7, 8, 9, 10])
def test_stocks_weekday_target_hours(day, hour):
    assert is_stocks_4h_target_hour(ny(2026, 7, day, hour))


@pytest.mark.parametrize(
    "value",
    [ny(2026, 7, 6, 8), ny(2026, 7, 6, 10), ny(2026, 7, 11, 9), ny(2026, 7, 12, 17)],
)
def test_stocks_other_hours_and_weekends_rejected(value):
    assert not is_stocks_4h_target_hour(value)


def test_retry_constants_and_bounded_delayed_starts():
    assert RETRY_MINUTES == {1, 21, 41}
    assert MAX_SCHEDULE_DELAY_MINUTES == 18
    for minute in (1, 19, 21, 39, 41, 59):
        assert is_retry_opportunity(ny(2026, 7, 6, 9, minute))
    for minute in (0, 20, 40):
        assert not is_retry_opportunity(ny(2026, 7, 6, 9, minute))


def test_complete_opportunity_requires_target_hour_and_retry_window():
    assert is_futures_4h_opportunity(ny(2026, 1, 6, 5, 19))
    assert not is_futures_4h_opportunity(ny(2026, 1, 6, 5, 20))
    assert is_stocks_4h_opportunity(ny(2026, 7, 6, 13, 59))
    assert not is_stocks_4h_opportunity(ny(2026, 7, 6, 14, 1))


@pytest.mark.parametrize(
    ("utc_value", "local_hour"),
    [
        (datetime(2026, 3, 8, 1, 1, tzinfo=timezone.utc), 20),  # before spring shift
        (datetime(2026, 3, 8, 7, 1, tzinfo=timezone.utc), 3),  # after spring shift
        (datetime(2026, 11, 1, 5, 1, tzinfo=timezone.utc), 1),  # first fall 01
        (datetime(2026, 11, 1, 6, 1, tzinfo=timezone.utc), 1),  # repeated fall 01
    ],
)
def test_dst_conversion_uses_new_york_wall_clock(utc_value, local_hour):
    assert utc_value.astimezone(NY).hour == local_hour
    assert is_retry_opportunity(utc_value)


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        is_retry_opportunity(datetime(2026, 1, 6, 9, 1))
