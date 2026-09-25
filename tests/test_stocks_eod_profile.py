from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / ".github/scripts/run_stocks_eod_guarded.py"
    spec = importlib.util.spec_from_file_location("run_stocks_eod_guarded_b1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_routes_once_to_family_and_preserves_downstream_order(monkeypatch):
    runner = load_runner()
    events = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, check: events.append(("command", command))
        or subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(runner, "run_combo_health",
                        lambda **kwargs: events.append(("health", kwargs["combos"][0])) or [])
    monkeypatch.setattr(runner, "print_results", lambda results: events.append(("print", results)))
    monkeypatch.setattr(
        runner, "notify_combo_signals",
        lambda combo, **kwargs: events.append(("notify", combo)),
    )

    runner.run_profile()

    commands = [event[1] for event in events if event[0] == "command"]
    assert commands == [
        [runner.sys.executable, str(ROOT / "jobs/run_stock_eod_family.py")],
        [runner.sys.executable, str(ROOT / "jobs/run_etf_trends.py"), "weekly"],
        [runner.sys.executable, str(ROOT / "jobs/run_etf_trends.py"), "daily"],
        [runner.sys.executable, str(ROOT / "jobs/run_combo.py"), "stocks", "stocks_c_dwm_shortlist"],
        [runner.sys.executable, str(ROOT / "jobs/run_combo.py"), "stocks", "stocks_c_dwm_all"],
    ]
    assert not any("run_timeframe.py" in " ".join(command) for command in commands)
    assert [event for event in events if event[0] == "health"] == [
        ("health", "stocks_c_dwm_shortlist"), ("health", "stocks_c_dwm_all")
    ]
    assert [event for event in events if event[0] == "notify"] == [
        ("notify", "stocks_c_dwm_shortlist"), ("notify", "stocks_c_dwm_all")
    ]


def test_family_subprocess_failure_propagates_and_stops_profile(monkeypatch):
    runner = load_runner()
    commands = []

    def fail(command, check):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    monkeypatch.setattr(runner, "notify_combo_signals",
                        lambda *args, **kwargs: pytest.fail("notification called"))
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_profile()
    assert commands == [[runner.sys.executable, str(ROOT / "jobs/run_stock_eod_family.py")]]

