from __future__ import annotations

# core/guard.py

from dataclasses import dataclass
from datetime import datetime, date, time
from pathlib import Path
from typing import Callable, Optional, Literal, Any, Dict

from zoneinfo import ZoneInfo
import pandas as pd

from core.paths import DATA
from core import storage

# -----------------------------
# Time helpers
# -----------------------------

NY_TZ = ZoneInfo("America/New_York")

def now_ny() -> datetime:
    return datetime.now(NY_TZ)

def minutes_since_midnight(t: time) -> int:
    return t.hour * 60 + t.minute

def _fmt_dt(dt: datetime) -> str:
    # Stable, filename-safe-ish timestamp (no ":" or "+")
    return dt.strftime("%Y-%m-%dT%H-%M-%S%z")

WindowMode = Literal["after_only", "abs", "range"]


@dataclass(frozen=True)
class WindowDecision:
    ok: bool
    reason: str


def check_time_window(
    *,
    now: datetime,
    target_time: time,
    tolerance_min: int,
    mode: WindowMode = "after_only",
    range_start: Optional[time] = None,
    range_end: Optional[time] = None,
) -> WindowDecision:
    """
    Determine if a job should run based on the current time in a timezone-aware datetime.

    mode:
      - "after_only": allow runs only AFTER target_time, and only within tolerance.
      - "abs": allow runs within +/- tolerance around target_time.
      - "range": allow runs between range_start and range_end inclusive.

    Notes:
      - now MUST be tz-aware (we use NY_TZ in callers).
      - tolerance_min should be >= 0
    """
    if tolerance_min < 0:
        return WindowDecision(False, "invalid tolerance_min < 0")

    now_t = now.time()
    now_min = minutes_since_midnight(now_t)
    target_min = minutes_since_midnight(target_time)

    if mode == "range":
        if range_start is None or range_end is None:
            return WindowDecision(False, "range mode requires range_start and range_end")
        start_min = minutes_since_midnight(range_start)
        end_min = minutes_since_midnight(range_end)
        if start_min <= now_min <= end_min:
            return WindowDecision(True, f"in range {range_start}..{range_end}")
        return WindowDecision(False, f"outside range {range_start}..{range_end}")

    if mode == "abs":
        diff = abs(now_min - target_min)
        if diff <= tolerance_min:
            return WindowDecision(True, f"within +/-{tolerance_min} min of target {target_time}")
        return WindowDecision(False, f"outside +/-{tolerance_min} min of target {target_time}")

    # mode == "after_only"
    diff = now_min - target_min
    if diff < 0:
        return WindowDecision(False, f"before target {target_time}")
    if diff <= tolerance_min:
        return WindowDecision(True, f"within {tolerance_min} min after target {target_time}")
    return WindowDecision(False, f"more than {tolerance_min} min after target {target_time}")


# -----------------------------
# Run keys (idempotency)
# -----------------------------

RunPeriod = Literal["daily", "weekly", "monthly", "custom"]


def make_run_key(*, period: RunPeriod, now: datetime, custom: Optional[str] = None) -> str:
    """
    Produce a run_key string scoped to NY time.

    daily:   YYYY-MM-DD
    weekly:  YYYY-Www (ISO week) e.g. 2025-W50
    monthly: YYYY-MM
    custom:  caller provides custom string
    """
    local_date: date = now.astimezone(NY_TZ).date()

    if period == "daily":
        return local_date.strftime("%Y-%m-%d")

    if period == "weekly":
        iso_year, iso_week, _ = local_date.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"

    if period == "monthly":
        return local_date.strftime("%Y-%m")

    if period == "custom":
        if not custom or not str(custom).strip():
            raise ValueError("custom period requires non-empty custom run key")
        # minimal normalization
        return str(custom).strip()

    raise ValueError(f"Unsupported period: {period!r}")


# -----------------------------
# Marker storage (Parquet)
# -----------------------------

def _marker_path(marker_name: str, run_key: str) -> Path:
    """
    Store markers under DATA/_runs/<marker_name>/<run_key>.parquet
    """
    safe_marker = str(marker_name).strip().replace(" ", "_")
    safe_key = str(run_key).strip().replace(" ", "_")
    return DATA / "_runs" / safe_marker / f"{safe_key}.parquet"


def has_run(marker_name: str, run_key: str) -> bool:
    """
    True if a marker exists for (marker_name, run_key).
    Uses core.storage.exists so this works on local and S3.
    """
    p = _marker_path(marker_name, run_key)
    return storage.exists(p)


def mark_run(
    marker_name: str,
    run_key: str,
    *,
    now: Optional[datetime] = None,
    status: str = "success",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a marker parquet for a successful run (or other status).
    """
    if now is None:
        now = now_ny()

    p = _marker_path(marker_name, run_key)

    row: Dict[str, Any] = {
        "marker": marker_name,
        "run_key": run_key,
        "status": status,
        "ran_at_ny": now.astimezone(NY_TZ).isoformat(),
        "ran_at_utc": now.astimezone(ZoneInfo("UTC")).isoformat(),
    }
    if meta:
        # Flatten only simple scalar values; stringify anything complex
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[f"meta_{k}"] = v
            else:
                row[f"meta_{k}"] = str(v)

    df = pd.DataFrame([row])

    # Ensure parent exists for local; for S3 the "dirs" are virtual (fine)
    # storage.save_parquet handles local mkdirs already.
    storage.save_parquet(df, p)


# -----------------------------
# One-call “guard then run”
# -----------------------------

def should_run(
    *,
    marker_name: str,
    period: RunPeriod,
    target_time: time,
    tolerance_min: int,
    mode: WindowMode,
    now: Optional[datetime] = None,
    custom_run_key: Optional[str] = None,
    respect_idempotency: bool = True,
    bypass_time_window: bool = False,
) -> tuple[bool, str, str]:
    """
    Returns (ok_to_run, run_key, reason).
    """
    if now is None:
        now = now_ny()

    run_key = make_run_key(period=period, now=now, custom=custom_run_key)

    if respect_idempotency and has_run(marker_name, run_key):
        return False, run_key, f"already ran for {run_key}"

    if not bypass_time_window:
        dec = check_time_window(
            now=now,
            target_time=target_time,
            tolerance_min=tolerance_min,
            mode=mode,
        )
        if not dec.ok:
            return False, run_key, dec.reason

    return True, run_key, "ok"


def run_guarded(
    *,
    marker_name: str,
    period: RunPeriod,
    target_time: time,
    tolerance_min: int,
    mode: WindowMode,
    fn: Callable[[], None],
    now: Optional[datetime] = None,
    custom_run_key: Optional[str] = None,
    respect_idempotency: bool = True,
    bypass_time_window: bool = False,
    mark_on_success: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Evaluate guard + optionally run fn() + write marker on success.

    Typical usage from guard scripts:
      - scheduled: bypass_time_window=False, respect_idempotency=True
      - manual:    bypass_time_window=True,  respect_idempotency=False (or True if you prefer)
    """
    ok, run_key, reason = should_run(
        marker_name=marker_name,
        period=period,
        target_time=target_time,
        tolerance_min=tolerance_min,
        mode=mode,
        now=now,
        custom_run_key=custom_run_key,
        respect_idempotency=respect_idempotency,
        bypass_time_window=bypass_time_window,
    )

    if not ok:
        print(f"[GUARD] skip {marker_name} ({run_key}): {reason}")
        return

    print(f"[GUARD] run {marker_name} ({run_key}): {reason}")
    fn()

    if mark_on_success:
        mark_run(marker_name, run_key, now=now_ny() if now is None else now, status="success", meta=meta)
        print(f"[GUARD] marked success {marker_name} ({run_key})")
