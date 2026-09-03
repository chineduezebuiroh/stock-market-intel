"""Deterministic target resolution and explicit censoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from .specs import MaturityStatus


@dataclass(frozen=True)
class HorizonResolution:
    target_date: date
    resolved_date: date | None
    status: MaturityStatus


def resolve_target_date(
    target: date,
    available_dates: Iterable[date],
    *,
    data_asof: date,
    tolerance_days: int = 7,
) -> HorizonResolution:
    """Resolve to the first observed session on/after target, never before it."""
    dates = sorted(set(available_dates))
    if not dates:
        return HorizonResolution(target, None, MaturityStatus.MISSING_PRICE_HISTORY)
    if data_asof < target:
        return HorizonResolution(target, None, MaturityStatus.IMMATURE)
    upper = target + timedelta(days=tolerance_days)
    resolved = next((value for value in dates if target <= value <= upper), None)
    if resolved is None:
        status = (
            MaturityStatus.IMMATURE
            if data_asof < upper
            else MaturityStatus.UNRESOLVABLE_TARGET_DATE
        )
        return HorizonResolution(target, None, status)
    return HorizonResolution(target, resolved, MaturityStatus.MATURE)
