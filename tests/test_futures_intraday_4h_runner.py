from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

from core import guard

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "run_futures_intraday_4h_guarded.py"
WORKFLOW = ROOT / ".github" / "workflows" / "futures_intraday_4h.yml"
NY = ZoneInfo("America/New_York")
REGISTRY_COLUMNS = ["job", "active", "last_execution", "check_window_hours"]


def load_runner():
    name = "run_futures_intraday_4h_guarded_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_runner()


def test_run_profile_has_only_the_4h_contract_in_order(runner, monkeypatch):
    events = []

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda cmd, check: events.append(("command", cmd, check)),
    )
    monkeypatch.setattr(
        runner,
        "run_combo_health",
        lambda **kwargs: events.append(("health", kwargs)) or ["healthy"],
    )
    monkeypatch.setattr(
        runner,
        "print_results",
        lambda results: events.append(("health_results", results)),
    )
    monkeypatch.setattr(
        runner,
        "notify_combo_signals",
        lambda *args, **kwargs: events.append(("notification", args, kwargs)),
    )

    runner.run_profile()

    commands = [event[1] for event in events if event[0] == "command"]
    assert commands == [
        [
            sys.executable,
            str(ROOT / "jobs" / "run_timeframe.py"),
            "futures",
            "intraday_1h",
            "--cascade",
        ],
        [
            sys.executable,
            str(ROOT / "jobs" / "run_timeframe.py"),
            "futures",
            "weekly",
        ],
        [
            sys.executable,
            str(ROOT / "jobs" / "run_combo.py"),
            "futures",
            "futures_2_4hdw_shortlist",
        ],
    ]
    assert [event[0] for event in events] == [
        "command",
        "command",
        "command",
        "health",
        "health_results",
        "notification",
    ]
    assert events[3][1] == {
        "combos": ["futures_2_4hdw_shortlist"],
        "universe_csv": "shortlist_futures.csv",
    }
    assert events[-1] == (
        "notification",
        ("futures_2_4hdw_shortlist",),
        {"only_if_changed": True},
    )
    assert not any(
        "futures_1_1h4hd_shortlist" in str(part)
        for event in events
        for part in event[1:]
    )


def test_manual_dispatch_bypasses_timing_and_registry(runner, monkeypatch):
    observed = {}
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setattr(
        runner,
        "now_ny",
        lambda: datetime(2026, 9, 19, 12, 34, tzinfo=NY),  # Saturday
    )
    monkeypatch.setattr(
        runner,
        "in_futures_session",
        lambda now: (_ for _ in ()).throw(AssertionError("session gate called")),
    )
    monkeypatch.setattr(
        runner,
        "near_4h_grid",
        lambda now: (_ for _ in ()).throw(AssertionError("grid gate called")),
    )
    monkeypatch.setattr(
        runner, "run_registry_guarded", lambda **kwargs: observed.update(kwargs) or True
    )

    runner.main()

    assert observed["job_name"] == "futures_intraday_4h"
    assert observed["fn"] is runner.run_profile
    assert observed["bypass_registry"] is True


def test_scheduled_execution_uses_session_and_grid_guards(runner, monkeypatch):
    observed = {}
    now = datetime(2026, 9, 21, 9, 1, tzinfo=NY)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setattr(runner, "now_ny", lambda: now)
    monkeypatch.setattr(runner, "in_futures_session", lambda value: value is now)
    monkeypatch.setattr(runner, "near_4h_grid", lambda value: value is now)
    monkeypatch.setattr(
        runner, "run_registry_guarded", lambda **kwargs: observed.update(kwargs) or True
    )

    runner.main()

    assert observed == {
        "job_name": "futures_intraday_4h",
        "fn": runner.run_profile,
        "now": now,
        "bypass_registry": False,
    }


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 9, 19, 9, 1, tzinfo=NY),  # Saturday
        datetime(2026, 9, 21, 10, 1, tzinfo=NY),  # not a 4h target hour
    ],
)
def test_scheduled_execution_skips_outside_session_or_grid(runner, monkeypatch, now):
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setattr(runner, "now_ny", lambda: now)
    monkeypatch.setattr(
        runner,
        "run_registry_guarded",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("registry guard called")),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.main()

    assert exc_info.value.code == 0


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 9, 20, 21, 0, tzinfo=NY), True),
        (datetime(2026, 9, 20, 17, 1, tzinfo=NY), False),
        (datetime(2026, 9, 21, 1, 0, tzinfo=NY), True),
        (datetime(2026, 9, 21, 5, 59, tzinfo=NY), True),
        (datetime(2026, 9, 21, 6, 0, tzinfo=NY), False),
        (datetime(2026, 9, 25, 13, 1, tzinfo=NY), True),
        (datetime(2026, 9, 25, 17, 1, tzinfo=NY), False),
        (datetime(2026, 9, 26, 9, 1, tzinfo=NY), False),
    ],
)
def test_current_futures_4h_target_boundaries(runner, now, expected):
    assert runner.near_4h_grid(now) is expected


def registry_state(monkeypatch, *, active="Yes", last_execution=pd.NA):
    state = {
        "value": pd.DataFrame(
            [["futures_intraday_4h", active, last_execution, 3.4]],
            columns=REGISTRY_COLUMNS,
        )
    }
    monkeypatch.setattr(guard, "load_execution_registry", lambda: state["value"].copy())
    monkeypatch.setattr(
        guard, "save_execution_registry", lambda value: state.update(value=value.copy())
    )
    return state


@pytest.mark.parametrize(
    ("active", "last_execution"),
    [
        ("No", pd.NA),
        ("Yes", pd.Timestamp("2026-09-21T12:00:00Z")),
    ],
)
def test_inactive_or_recent_registry_skips_profile(
    runner, monkeypatch, active, last_execution
):
    state = registry_state(monkeypatch, active=active, last_execution=last_execution)
    monkeypatch.setattr(
        runner,
        "run_profile",
        lambda: (_ for _ in ()).throw(AssertionError("profile executed")),
    )

    executed = runner.run_registry_guarded(
        job_name=runner.JOB_NAME,
        fn=runner.run_profile,
        now=datetime(2026, 9, 21, 9, 1, tzinfo=NY),
    )

    assert executed is False
    if pd.isna(last_execution):
        assert pd.isna(state["value"].iloc[0]["last_execution"])
    else:
        assert state["value"].iloc[0]["last_execution"] == last_execution


@pytest.mark.parametrize(
    "failure_stage", ["prerequisite", "combo", "health", "notification"]
)
def test_profile_failure_prevents_later_work_and_registry_mark(
    runner, monkeypatch, failure_stage
):
    state = registry_state(monkeypatch)
    events = []
    command_number = 0 if failure_stage == "prerequisite" else 2

    def subprocess_run(cmd, check):
        events.append(("command", cmd))
        if (
            failure_stage in {"prerequisite", "combo"}
            and len(events) - 1 == command_number
        ):
            raise RuntimeError(failure_stage)

    def health(**kwargs):
        events.append(("health", kwargs))
        if failure_stage == "health":
            raise RuntimeError("health")
        return []

    def notify(*args, **kwargs):
        events.append(("notification", args, kwargs))
        if failure_stage == "notification":
            raise RuntimeError("notification")

    monkeypatch.setattr(runner.subprocess, "run", subprocess_run)
    monkeypatch.setattr(runner, "run_combo_health", health)
    monkeypatch.setattr(runner, "print_results", lambda results: None)
    monkeypatch.setattr(runner, "notify_combo_signals", notify)

    with pytest.raises(RuntimeError, match=failure_stage):
        runner.run_registry_guarded(
            job_name=runner.JOB_NAME,
            fn=runner.run_profile,
            now=datetime(2026, 9, 21, 9, 1, tzinfo=NY),
        )

    assert pd.isna(state["value"].iloc[0]["last_execution"])
    if failure_stage == "prerequisite":
        assert len(events) == 1


def test_success_marks_only_futures_4h(runner, monkeypatch):
    state = registry_state(monkeypatch)
    monkeypatch.setattr(runner, "run_profile", lambda: None)

    executed = runner.run_registry_guarded(
        job_name=runner.JOB_NAME,
        fn=runner.run_profile,
        now=datetime(2026, 9, 21, 9, 1, tzinfo=NY),
    )

    assert executed is True
    assert state["value"]["job"].tolist() == ["futures_intraday_4h"]
    assert pd.notna(state["value"].iloc[0]["last_execution"])


def test_workflow_remains_manual_only_with_canonical_s3_environment():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    triggers = workflow[True]
    env = workflow["jobs"]["futures_intraday_4h"]["env"]

    assert "schedule" not in triggers
    assert "workflow_dispatch" in triggers
    assert env == {
        "DATA_BACKEND": "s3",
        "S3_BUCKET_DATA": "stock-intel-data-prod",
        "S3_PREFIX_DATA": "data",
        "AWS_ACCESS_KEY_ID": "${{ secrets.AWS_ACCESS_KEY_ID }}",
        "AWS_SECRET_ACCESS_KEY": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
        "AWS_DEFAULT_REGION": "us-east-1",
        "TELEGRAM_BOT_TOKEN": "${{ secrets.TELEGRAM_BOT_TOKEN }}",
        "TELEGRAM_CHAT_ID": "${{ secrets.TELEGRAM_CHAT_ID }}",
    }
