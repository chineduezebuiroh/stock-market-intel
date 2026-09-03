"""Pure coverage, revision, corporate-action, and summary calculations."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .horizons import resolve_target_date
from .metrics import Direction, PriceBar, calculate_metrics, directional_return
from .specs import CoverageStatus, Horizon

REVISION_THRESHOLDS = {
    "over_1bp": 0.0001,
    "over_10bp": 0.001,
    "over_50bp": 0.005,
    "over_1pct": 0.01,
    "over_5pct": 0.05,
}


def revision_measures(immutable: float, rolling: float) -> dict[str, Any]:
    """Describe a rolling-vs-immutable value without changing either value."""
    valid = np.isfinite(immutable) and np.isfinite(rolling) and immutable > 0
    absolute = abs(rolling - immutable) if valid else np.nan
    percentage = absolute / immutable if valid else np.nan
    result: dict[str, Any] = {
        "absolute_difference": absolute,
        "percentage_difference": percentage,
        "exact_or_near_exact": bool(valid and percentage <= 1e-8),
    }
    result.update(
        {
            name: bool(valid and percentage > value)
            for name, value in REVISION_THRESHOLDS.items()
        }
    )
    return result


def corporate_action_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag adjustment-ratio breaks; never adjust or remove source prices.

    Primary flag: adjacent valid ``adj_close / close`` values differ by more than
    5%. Corroborating split-like flag: raw close moves by more than 40% while the
    adjustment ratio moves inversely by more than 20%. The primary criterion is
    intentionally conservative and can include large distributions.
    """
    columns = [
        "event_date",
        "adjustment_ratio",
        "ratio_change",
        "raw_close_change",
        "ratio_break_flag",
        "split_like_flag",
    ]
    if "adj_close" not in frame or "close" not in frame:
        return pd.DataFrame(columns=columns)
    close = pd.to_numeric(frame["close"], errors="coerce")
    adjusted = pd.to_numeric(frame["adj_close"], errors="coerce")
    ratio = adjusted / close.where(close > 0)
    ratio_change = ratio.pct_change(fill_method=None)
    raw_change = close.pct_change(fill_method=None)
    ratio_break = ratio_change.abs() > 0.05
    split_like = (
        (raw_change.abs() > 0.40)
        & (ratio_change.abs() > 0.20)
        & (np.sign(raw_change) != np.sign(ratio_change))
    )
    flagged = ratio_break | split_like
    return pd.DataFrame(
        {
            "event_date": frame.index,
            "adjustment_ratio": ratio,
            "ratio_change": ratio_change,
            "raw_close_change": raw_change,
            "ratio_break_flag": ratio_break,
            "split_like_flag": split_like,
        },
        index=frame.index,
    ).loc[flagged, columns]


def audit_horizon(
    *,
    direction: Direction,
    entry_date: date,
    immutable_entry_close: float,
    horizon: Horizon,
    frame: pd.DataFrame | None,
    dataset_asof: date,
    tolerance_days: int = 7,
) -> dict[str, Any]:
    """Audit terminal and full-path coverage while preserving entry truth."""
    target = entry_date + timedelta(days=horizon.calendar_days)
    base: dict[str, Any] = {
        "target_calendar_date": target,
        "theoretically_mature": target <= dataset_asof,
        "coverage_status": None,
        "terminal_covered": False,
        "path_covered": False,
        "resolved_exit_date": None,
        "exit_close": np.nan,
        "directional_return": np.nan,
        "mfe": np.nan,
        "mae": np.nan,
        "elapsed_calendar_days": pd.NA,
        "elapsed_trading_sessions": pd.NA,
    }
    if target > dataset_asof:
        return base | {"coverage_status": CoverageStatus.IMMATURE.value}
    if frame is None or frame.empty:
        return base | {"coverage_status": CoverageStatus.MISSING_SYMBOL_HISTORY.value}
    if not np.isfinite(immutable_entry_close) or immutable_entry_close <= 0:
        return base | {"coverage_status": CoverageStatus.INVALID_PRICE_DATA.value}

    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index).normalize()
    valid_close = pd.to_numeric(normalized.get("close"), errors="coerce")
    close_dates = [value.date() for value in normalized.index[valid_close.gt(0)]]
    resolution = resolve_target_date(
        target, close_dates, data_asof=dataset_asof, tolerance_days=tolerance_days
    )
    if resolution.resolved_date is None:
        first = normalized.index.min().date()
        status = (
            CoverageStatus.ENTRY_PREDATES_RETAINED_HISTORY
            if target < first
            else CoverageStatus.UNRESOLVABLE_TARGET_DATE
        )
        return base | {"coverage_status": status.value}

    exit_ts = pd.Timestamp(resolution.resolved_date)
    exit_close = pd.to_numeric(
        pd.Series([normalized.loc[exit_ts, "close"]]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(exit_close) or exit_close <= 0:
        return base | {"coverage_status": CoverageStatus.INVALID_PRICE_DATA.value}
    terminal = base | {
        "resolved_exit_date": resolution.resolved_date,
        "exit_close": float(exit_close),
        "directional_return": directional_return(
            direction, immutable_entry_close, float(exit_close)
        ),
        "terminal_covered": True,
        "elapsed_calendar_days": (resolution.resolved_date - entry_date).days,
    }

    window = normalized[
        (normalized.index.date > entry_date)
        & (normalized.index.date <= resolution.resolved_date)
    ]
    path_values = window.reindex(columns=["high", "low", "close"]).apply(
        pd.to_numeric, errors="coerce"
    )
    path_valid = (
        not path_values.empty
        and np.isfinite(path_values).all(axis=None)
        and (path_values > 0).all(axis=None)
    )
    retained_from_entry = normalized.index.min().date() <= entry_date
    if not path_valid:
        return terminal | {"coverage_status": CoverageStatus.INVALID_PRICE_DATA.value}
    if not retained_from_entry:
        return terminal | {
            "coverage_status": CoverageStatus.INCOMPLETE_EXCURSION_WINDOW.value
        }
    bars = [
        PriceBar(index.date(), row.high, row.low, row.close)
        for index, row in path_values.iterrows()
    ]
    metrics = calculate_metrics(
        direction, entry_date, immutable_entry_close, bars[-1], bars
    )
    return terminal | {
        "coverage_status": CoverageStatus.MATURE.value,
        "path_covered": True,
        "mfe": metrics.mfe,
        "mae": metrics.mae,
        "elapsed_trading_sessions": metrics.elapsed_trading_sessions,
    }


def summarize_coverage(observations: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Use theoretically mature observations as both coverage denominators."""
    rows = []
    for keys, part in observations.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mature = part[part["theoretically_mature"]]
        statuses = part["coverage_status"]
        denominator = len(mature)
        rows.append(
            dict(zip(groups, keys))
            | {
                "total_preparticipation_opportunities": len(part),
                "theoretically_mature_count": denominator,
                "mature_terminal_return_count": int(part["terminal_covered"].sum()),
                "complete_path_count": int(part["path_covered"].sum()),
                "immature_count": int(statuses.eq(CoverageStatus.IMMATURE.value).sum()),
                "missing_symbol_count": int(
                    statuses.eq(CoverageStatus.MISSING_SYMBOL_HISTORY.value).sum()
                ),
                "retention_truncated_count": int(
                    statuses.isin(
                        [
                            CoverageStatus.ENTRY_PREDATES_RETAINED_HISTORY.value,
                            CoverageStatus.INCOMPLETE_EXCURSION_WINDOW.value,
                        ]
                    ).sum()
                ),
                "unresolvable_count": int(
                    statuses.eq(CoverageStatus.UNRESOLVABLE_TARGET_DATE.value).sum()
                ),
                "invalid_price_count": int(
                    statuses.eq(CoverageStatus.INVALID_PRICE_DATA.value).sum()
                ),
                "corporate_action_flagged_count": int(
                    part["corporate_action_flag"].sum()
                ),
                "terminal_coverage_of_theoretically_mature": (
                    part["terminal_covered"].sum() / denominator
                    if denominator
                    else np.nan
                ),
                "path_coverage_of_theoretically_mature": (
                    part["path_covered"].sum() / denominator if denominator else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
