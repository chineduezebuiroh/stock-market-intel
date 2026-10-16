from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import json
from core.health import stock_eod_final_coverage
import pandas as pd

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
    def run(command, check):
        events.append(("command", command))
        if "--coverage-report" in command:
            Path(command[-1]).write_text(json.dumps(example_report()))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "notify_stock_eod_failure", lambda *args: pytest.fail("failure notice"))
    monkeypatch.setattr(runner, "run_combo_health",
                        lambda **kwargs: events.append(("health", kwargs["combos"][0])) or [])
    monkeypatch.setattr(runner, "print_results", lambda results: events.append(("print", results)))
    monkeypatch.setattr(
        runner, "notify_combo_signals",
        lambda combo, **kwargs: events.append(("notify", combo)),
    )

    runner.run_profile()

    commands = [event[1] for event in events if event[0] == "command"]
    assert commands[0][2] == "--coverage-report"
    assert commands[0][3].endswith("coverage.json")
    commands[0] = commands[0][:2]
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
    notices = []
    monkeypatch.setattr(runner, "notify_stock_eod_failure", lambda *args: notices.append(args))
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_profile()
    assert len(commands) == 1
    assert commands[0][:2] == [runner.sys.executable, str(ROOT / "jobs/run_stock_eod_family.py")]
    assert len(notices) == 1



def example_report(fail=False):
    symbols = [str(i) for i in range(10)]
    snapshots = {tf: pd.DataFrame({"symbol": symbols[:8] if fail and tf == "weekly" else symbols})
                 for tf in ("daily", "weekly", "monthly")}
    report = stock_eod_final_coverage("synthetic-family", snapshots, symbols)
    report["shortlist_missing"] = ["REQUIRED"] if fail else []
    return report


@pytest.mark.parametrize("failure", ["coverage", "health", "combo", "mirror", "signal"])
@pytest.mark.parametrize("notification_fails", [False, True])
def test_failure_notice_attempt_preserves_original_and_current_invocation(monkeypatch, failure, notification_fails, capsys):
    runner = load_runner()
    report = example_report(failure == "coverage")
    original = RuntimeError("original " + failure)
    events = []
    def run(command, check):
        events.append("command")
        if "--coverage-report" in command:
            Path(command[-1]).write_text(json.dumps(report))
            if failure == "mirror":
                raise original
        if failure == "combo" and "run_combo.py" in " ".join(command):
            raise original
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    from core.health import HealthResult
    monkeypatch.setattr(runner, "run_combo_health", lambda **kw: [HealthResult(False, "shortlist", "missing REQUIRED")] if failure == "health" else [])
    def health(results):
        if failure == "health":
            raise original
    monkeypatch.setattr(runner, "print_results", health)
    def signal(*args, **kwargs):
        events.append("signal")
        if failure == "signal":
            raise original
    monkeypatch.setattr(runner, "notify_combo_signals", signal)
    notices = []
    def notify(*args):
        notices.append(args)
        if notification_fails:
            raise OSError("delivery unavailable")
    monkeypatch.setattr(runner, "notify_stock_eod_failure", notify)
    with pytest.raises(RuntimeError) as caught:
        runner.run_profile()
    if failure != "coverage":
        assert caught.value is original
    else:
        assert "catastrophic coverage" in str(caught.value)
        assert events.count("command") == 5  # no early coverage abort
        assert "signal" not in events
    assert len(notices) == 1 and notices[0][1] == report
    if failure == "health":
        assert "missing REQUIRED" in notices[0][2][0]
        assert "signal" not in events
    if notification_fails:
        assert "original failure remains primary" in capsys.readouterr().out
