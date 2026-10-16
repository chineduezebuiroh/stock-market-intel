"""Run-scoped, best-effort wall timing; never a production policy input."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import time


TIMEFRAMES = ("daily", "weekly", "monthly")
_ACTIVE = ContextVar("stock_eod_performance", default=None)


class FamilyPerformance:
    def __init__(self):
        self.family_run_id = None
        self.timing_errors = 0
        self.bins = {
            tf: {name: {"count": 0, "total_seconds": 0.0} for name in (
                "processing", "storage_exists", "storage_read", "storage_write",
                "indicator", "snapshot",
            )}
            for tf in TIMEFRAMES
        }

    def summary(self):
        processing, snapshot = {}, {}
        for tf, bins in self.bins.items():
            total = bins["processing"]["total_seconds"]
            storage = sum(bins[name]["total_seconds"] for name in (
                "storage_exists", "storage_read", "storage_write",
            ))
            indicator = bins["indicator"]["total_seconds"]
            processing[tf] = {
                "count": bins["processing"]["count"],
                "total_seconds": total,
                "storage_seconds": storage,
                "indicator_seconds": indicator,
                # Nested storage/indicator spans are disjoint, inside processing.
                "residual_seconds": max(0.0, total - storage - indicator),
                "storage": {
                    name: dict(bins[f"storage_{name}"])
                    for name in ("exists", "read", "write")
                },
            }
            snapshot[tf] = dict(bins["snapshot"])
        return {
            "schema_version": 1, "family_run_id": self.family_run_id,
            "timing_errors": self.timing_errors,
            "processing": processing, "snapshot": snapshot,
        }


@contextmanager
def measure(timeframe, category):
    collector = _ACTIVE.get()
    started = None
    if collector is not None:
        try:
            collector.bins[timeframe][category]["count"] += 1
            started = time.perf_counter()
        except Exception:
            collector.timing_errors += 1
    try:
        yield
    finally:
        if collector is not None and started is not None:
            try:
                collector.bins[timeframe][category]["total_seconds"] += (
                    time.perf_counter() - started
                )
            except Exception:
                collector.timing_errors += 1


def timed_stage(category):
    """Preserve callable outcomes, including early returns and exceptions."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(namespace, timeframe, *args, **kwargs):
            if namespace != "stocks" or timeframe not in TIMEFRAMES:
                return fn(namespace, timeframe, *args, **kwargs)
            with measure(timeframe, category):
                return fn(namespace, timeframe, *args, **kwargs)
        return wrapped
    return decorate


def identify_family(run_id):
    try:
        collector = _ACTIVE.get()
        if collector is not None:
            collector.family_run_id = run_id
    except Exception:
        pass


def family_performance(fn):
    """Emit one bounded summary on success or failure; restore scope on exit."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            collector = FamilyPerformance()
            token = _ACTIVE.set(collector)
        except Exception:
            return fn(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        finally:
            try:
                _ACTIVE.reset(token)
            except Exception:
                pass
            try:
                print("[STOCK_EOD_PERF] " + json.dumps(
                    collector.summary(), sort_keys=True, allow_nan=False,
                ), flush=True)
            except Exception:
                # Diagnostic timing/serialization/output must never fail the job
                # or replace the original processing/publication exception.
                pass
    return wrapped
