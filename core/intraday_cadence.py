"""Pure New York-local cadence policy for governed intraday profiles."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

RETRY_MINUTES = frozenset({1, 21, 41})
MAX_SCHEDULE_DELAY_MINUTES = 18
FUTURES_4H_TARGET_HOURS = frozenset({1, 5, 9, 13, 17, 21})
STOCKS_4H_TARGET_HOURS = frozenset({9, 13, 17})


def as_new_york(now: datetime) -> datetime:
    """Return an aware datetime in New York; reject ambiguous naive inputs."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("intraday cadence requires a timezone-aware datetime")
    return now.astimezone(NY_TZ)


def is_retry_opportunity(now: datetime) -> bool:
    """Accept a nominal retry minute and at most 18 minutes of runner delay.

    The resulting windows are :01-:19, :21-:39, and :41-:59.  They never
    cross an hour boundary, so a delayed invocation cannot acquire the next
    hour's market eligibility.
    """
    minute = as_new_york(now).minute
    return any(
        retry_minute <= minute <= retry_minute + MAX_SCHEDULE_DELAY_MINUTES
        for retry_minute in RETRY_MINUTES
    )


def is_futures_4h_target_hour(now: datetime) -> bool:
    """Apply the 4h target-hour policy within the approximate futures week."""
    local = as_new_york(now)
    weekday = local.weekday()

    if weekday == 6:  # Sunday evening session
        return local.hour == 21
    if weekday <= 3:  # Monday through Thursday
        return local.hour in FUTURES_4H_TARGET_HOURS
    if weekday == 4:  # Friday session closes before the 17:00 target
        return local.hour in {1, 5, 9, 13}
    return False


def is_stocks_4h_target_hour(now: datetime) -> bool:
    """Apply the weekday 09/13/17 ET Stocks 4h target-hour policy."""
    local = as_new_york(now)
    return local.weekday() <= 4 and local.hour in STOCKS_4H_TARGET_HOURS


def is_futures_4h_opportunity(now: datetime) -> bool:
    return is_futures_4h_target_hour(now) and is_retry_opportunity(now)


def is_stocks_4h_opportunity(now: datetime) -> bool:
    return is_stocks_4h_target_hour(now) and is_retry_opportunity(now)
