from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core import guard

NOW = datetime(2026, 9, 20, 3, 0, tzinfo=ZoneInfo("America/New_York"))
COLUMNS = ["job", "active", "last_execution", "check_window_hours"]


def registry_row(*, active="Yes", last_execution=pd.NA, window=24):
    return pd.DataFrame(
        [["weekly_build_options_universe", active, last_execution, window]],
        columns=COLUMNS,
    )


@pytest.fixture
def registry(monkeypatch):
    state = {"value": pd.DataFrame(columns=COLUMNS)}
    monkeypatch.setattr(guard, "load_execution_registry", lambda: state["value"].copy())
    monkeypatch.setattr(
        guard, "save_execution_registry", lambda value: state.update(value=value.copy())
    )
    return state


def test_missing_known_job_bootstraps_and_is_persisted_after_success(registry):
    calls = []

    executed = guard.run_registry_guarded(
        job_name="weekly_build_options_universe",
        fn=lambda: calls.append("ran"),
        now=NOW,
    )

    assert executed is True
    assert calls == ["ran"]
    row = registry["value"].iloc[0]
    assert row["job"] == "weekly_build_options_universe"
    assert row["active"] == "Yes"
    assert row["check_window_hours"] == 24
    assert pd.Timestamp(row["last_execution"]) == pd.Timestamp(NOW).tz_convert("UTC")


@pytest.mark.parametrize(
    ("row", "expected_reason"),
    [
        (registry_row(active="No"), "inactive"),
        (
            registry_row(last_execution=NOW - timedelta(hours=2)),
            "last execution 2.00h ago <",
        ),
    ],
)
def test_persisted_job_can_be_inactive_or_recent(registry, row, expected_reason):
    registry["value"] = row

    ok, reason = guard.should_run_from_registry(
        job_name="weekly_build_options_universe", now=NOW
    )

    assert ok is False
    assert expected_reason in reason


def test_eligible_persisted_job_runs_without_overwriting_configuration(registry):
    registry["value"] = registry_row(
        active="Yes", last_execution=NOW - timedelta(hours=25), window=23
    )

    executed = guard.run_registry_guarded(
        job_name="weekly_build_options_universe", fn=lambda: None, now=NOW
    )

    assert executed is True
    row = registry["value"].iloc[0]
    assert row["active"] == "Yes"
    assert row["check_window_hours"] == 23


def test_unknown_missing_job_fails_closed(registry):
    called = False

    def run():
        nonlocal called
        called = True

    executed = guard.run_registry_guarded(job_name="unknown_job", fn=run, now=NOW)

    assert executed is False
    assert called is False
    assert registry["value"].empty


def test_manual_bypass_runs_and_marks_even_when_persisted_job_is_inactive(registry):
    registry["value"] = registry_row(active="No")
    calls = []

    executed = guard.run_registry_guarded(
        job_name="weekly_build_options_universe",
        fn=lambda: calls.append("ran"),
        now=NOW,
        bypass_registry=True,
    )

    assert executed is True
    assert calls == ["ran"]
    assert registry["value"].iloc[0]["active"] == "No"


def test_failure_propagates_and_is_not_marked(registry):
    def fail():
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        guard.run_registry_guarded(
            job_name="weekly_build_options_universe", fn=fail, now=NOW
        )

    assert registry["value"].empty
