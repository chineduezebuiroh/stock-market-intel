from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import pandas as pd

from core import guard
from core.guard import JOB_REGISTRY_DEFAULTS

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts"
NY = ZoneInfo("America/New_York")


def load_runner(stem):
    name = f"{stem}_contract_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hourly():
    return load_runner("run_futures_intraday_1h_guarded")


@pytest.fixture
def stocks():
    return load_runner("run_stocks_intraday_4h_guarded")


def test_registry_defaults_remain_governed_per_profile():
    assert JOB_REGISTRY_DEFAULTS["futures_intraday_4h"] == {
        "active": "Yes",
        "check_window_hours": 3.4,
    }
    assert JOB_REGISTRY_DEFAULTS["stocks_intraday_4h"] == {
        "active": "Yes",
        "check_window_hours": 3.4,
    }
    assert JOB_REGISTRY_DEFAULTS["futures_intraday_1h"] == {
        "active": "No",
        "check_window_hours": 0.85,
    }


def test_hourly_profile_contract(hourly, monkeypatch):
    events = []
    monkeypatch.setattr(
        hourly.subprocess,
        "run",
        lambda command, check: events.append(("command", command, check)),
    )
    monkeypatch.setattr(hourly, "_env_flag", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        hourly,
        "run_combo_health",
        lambda **kwargs: events.append(("health", kwargs)) or [],
    )
    monkeypatch.setattr(hourly, "print_results", lambda results: None)
    monkeypatch.setattr(
        hourly,
        "notify_combo_signals",
        lambda *args, **kwargs: events.append(("notification", args, kwargs)),
    )

    hourly.run_profile()

    commands = [event[1] for event in events if event[0] == "command"]
    assert commands == [
        [
            sys.executable,
            str(ROOT / "jobs/run_timeframe.py"),
            "futures",
            "intraday_1h",
            "--cascade",
        ],
        [
            sys.executable,
            str(ROOT / "jobs/run_combo.py"),
            "futures",
            "futures_1_1h4hd_shortlist",
        ],
    ]
    assert events[-1] == (
        "notification",
        ("futures_1_1h4hd_shortlist",),
        {"only_if_changed": True},
    )


def test_hourly_scheduled_and_manual_registry_behavior(hourly, monkeypatch):
    now = datetime(2026, 9, 21, 9, 1, tzinfo=NY)  # also a 4h target
    calls = []
    monkeypatch.setattr(hourly, "now_ny", lambda: now)
    monkeypatch.setattr(hourly, "in_futures_session", lambda value: True)
    monkeypatch.setattr(hourly, "is_retry_opportunity", lambda value: True)
    monkeypatch.setattr(
        hourly,
        "run_registry_guarded",
        lambda **kwargs: calls.append(kwargs) or False,
    )

    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    hourly.main()
    assert calls[-1]["job_name"] == "futures_intraday_1h"
    assert calls[-1]["bypass_registry"] is False

    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    hourly.main()
    assert calls[-1]["job_name"] == "futures_intraday_1h"
    assert calls[-1]["bypass_registry"] is True


def test_hourly_inactive_skip_manual_bypass_and_retry_window(monkeypatch):
    columns = ["job", "active", "last_execution", "check_window_hours"]
    state = {
        "value": pd.DataFrame(
            [["futures_intraday_1h", "No", pd.NA, 0.85]], columns=columns
        )
    }
    monkeypatch.setattr(guard, "load_execution_registry", lambda: state["value"].copy())
    monkeypatch.setattr(
        guard, "save_execution_registry", lambda value: state.update(value=value.copy())
    )
    calls = []
    at_0901 = datetime(2026, 9, 21, 9, 1, tzinfo=NY)

    assert not guard.run_registry_guarded(
        job_name="futures_intraday_1h", fn=lambda: calls.append("run"), now=at_0901
    )
    assert calls == []
    assert guard.run_registry_guarded(
        job_name="futures_intraday_1h",
        fn=lambda: calls.append("manual"),
        now=at_0901,
        bypass_registry=True,
    )
    assert calls == ["manual"]

    state["value"].loc[0, ["active", "last_execution"]] = ["Yes", pd.NaT]
    assert guard.run_registry_guarded(
        job_name="futures_intraday_1h", fn=lambda: calls.append("09:01"), now=at_0901
    )
    for minute in (21, 41):
        assert not guard.run_registry_guarded(
            job_name="futures_intraday_1h",
            fn=lambda: calls.append("unexpected"),
            now=at_0901.replace(minute=minute),
        )
    assert guard.run_registry_guarded(
        job_name="futures_intraday_1h",
        fn=lambda: calls.append("10:01"),
        now=at_0901.replace(hour=10),
    )
    assert calls == ["manual", "09:01", "10:01"]


def test_stocks_profile_contract(stocks, monkeypatch):
    events = []
    monkeypatch.setattr(
        stocks.subprocess,
        "run",
        lambda command, check: events.append(("command", command, check)),
    )
    monkeypatch.setattr(stocks.storage, "exists", lambda path: False)
    monkeypatch.setattr(
        stocks,
        "run_combo_health",
        lambda **kwargs: events.append(("health", kwargs)) or [],
    )
    monkeypatch.setattr(stocks, "print_results", lambda results: None)
    monkeypatch.setattr(
        stocks,
        "notify_combo_signals",
        lambda *args, **kwargs: events.append(("notification", args, kwargs)),
    )

    stocks.run_profile()

    commands = [event[1] for event in events if event[0] == "command"]
    assert commands[0][2:] == [
        "stocks",
        "intraday_4h",
        "--cascade",
        "--allowed-universes",
        "shortlist_stocks",
    ]
    assert commands[1][2:] == ["stocks", "stocks_d_4hdw_shortlist"]
    assert events[-1] == (
        "notification",
        ("stocks_d_4hdw_shortlist",),
        {"only_if_changed": True},
    )
