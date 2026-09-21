from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"
SCRIPT = SCRIPTS / "run_futures_intraday_orchestrator.py"
NY = ZoneInfo("America/New_York")


def load_orchestrator():
    sys.path.insert(0, str(SCRIPTS))
    try:
        name = "run_futures_intraday_orchestrator_test"
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_4h_branch_calls_only_self_contained_4h_profile(monkeypatch):
    orchestrator = load_orchestrator()
    now = datetime(2026, 9, 21, 9, 1, tzinfo=NY)
    calls = []
    monkeypatch.setattr(orchestrator, "now_ny", lambda: now)
    monkeypatch.setattr(orchestrator.g1h, "in_futures_session", lambda value: True)
    monkeypatch.setattr(orchestrator.g1h, "near_hour_plus_one", lambda value: True)
    monkeypatch.setattr(orchestrator.g4h, "near_4h_grid", lambda value: True)
    monkeypatch.setattr(orchestrator.g4h, "in_futures_session", lambda value: True)
    monkeypatch.setattr(
        orchestrator.g1h,
        "run_profile",
        lambda: (_ for _ in ()).throw(AssertionError("1h profile called")),
    )
    monkeypatch.setattr(orchestrator.g4h, "run_profile", lambda: calls.append("4h"))
    monkeypatch.setattr(
        orchestrator,
        "should_run_from_registry",
        lambda **kwargs: (True, "eligible"),
    )
    monkeypatch.setattr(
        orchestrator,
        "mark_registry_execution",
        lambda **kwargs: calls.append(("mark", kwargs["job_name"])),
    )

    orchestrator._run_if_ready()

    assert calls == ["4h", ("mark", "futures_intraday_4h")]


def test_non_4h_branch_preserves_standalone_1h_behavior(monkeypatch):
    orchestrator = load_orchestrator()
    now = datetime(2026, 9, 21, 10, 1, tzinfo=NY)
    calls = []
    monkeypatch.setattr(orchestrator, "now_ny", lambda: now)
    monkeypatch.setattr(orchestrator.g1h, "in_futures_session", lambda value: True)
    monkeypatch.setattr(orchestrator.g1h, "near_hour_plus_one", lambda value: True)
    monkeypatch.setattr(orchestrator.g4h, "near_4h_grid", lambda value: False)
    monkeypatch.setattr(orchestrator.g4h, "in_futures_session", lambda value: True)
    monkeypatch.setattr(orchestrator.g1h, "run_profile", lambda: calls.append("1h"))
    monkeypatch.setattr(
        orchestrator.g4h,
        "run_profile",
        lambda: (_ for _ in ()).throw(AssertionError("4h profile called")),
    )
    monkeypatch.setattr(
        orchestrator,
        "should_run_from_registry",
        lambda **kwargs: (True, "eligible"),
    )
    monkeypatch.setattr(
        orchestrator,
        "mark_registry_execution",
        lambda **kwargs: calls.append(("mark", kwargs["job_name"])),
    )

    orchestrator._run_if_ready()

    assert calls == ["1h", ("mark", "futures_intraday_1h")]
