"""Resolve immutable lower-bar closes to their represented daily session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

LOWER_FREQUENCY = {
    "stocks_c_dwm_all": "DAILY",
    "stocks_b_wmq_all": "WEEKLY",
    "stocks_a_mqy_all": "MONTHLY",
}


@dataclass(frozen=True)
class EffectiveEntryDate:
    date: date | None
    method: str


def _execution_market_date(value: object) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("America/New_York").date()


def resolve_effective_entry_date(
    *,
    combo: str,
    lower_bar_date: date,
    execution_timestamp: object,
    frame: pd.DataFrame,
) -> EffectiveEntryDate:
    """Return the last valid daily close in the labelled lower bar by execution.

    The execution timestamp, rather than wall-clock time, caps the available daily
    calendar.  A result is never allowed to escape the labelled day/week/month.
    """
    frequency = LOWER_FREQUENCY.get(combo)
    if frequency is None:
        return EffectiveEntryDate(None, "UNSUPPORTED_COMBO")
    start = pd.Timestamp(lower_bar_date).normalize()
    if frequency == "DAILY":
        end = start
    elif frequency == "WEEKLY":
        end = start + pd.Timedelta(days=6)
    else:
        end = start + pd.offsets.MonthEnd(0)
    cap = min(end, pd.Timestamp(_execution_market_date(execution_timestamp)))
    if cap < start:
        return EffectiveEntryDate(None, f"{frequency}_NO_VALID_DAILY_SESSION")
    # A daily lower label already is the represented market session.  Preserve
    # that semantic even when the separate outcome-price history is missing.
    if frequency == "DAILY":
        return EffectiveEntryDate(start.date(), "DAILY_LOWER_BAR_DATE")
    if frame is None or frame.empty or "close" not in frame:
        return EffectiveEntryDate(None, f"{frequency}_NO_VALID_DAILY_SESSION")

    dates = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
    valid = pd.DataFrame({"date": dates, "close": closes})
    valid = valid[valid.close.gt(0) & valid.date.ge(start) & valid.date.le(cap)]
    if valid.empty:
        return EffectiveEntryDate(None, f"{frequency}_NO_VALID_DAILY_SESSION")
    resolved = valid.date.max().date()
    if not start.date() <= resolved <= end.date():
        return EffectiveEntryDate(None, f"{frequency}_OUTSIDE_LOWER_BAR")
    return EffectiveEntryDate(resolved, f"{frequency}_LATEST_AVAILABLE_DAILY_SESSION")
