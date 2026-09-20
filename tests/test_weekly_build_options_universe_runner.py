import importlib.util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / ".github/scripts/run_weekly_build_options_universe.py"
WORKFLOW = ROOT / ".github/workflows/weekly_build_options_universe.yml"
SPEC = importlib.util.spec_from_file_location("weekly_options_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_workflow_uses_canonical_production_registry_backend():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    env = workflow["jobs"]["build-options-universe"]["env"]

    assert env["DATA_BACKEND"] == "s3"
    assert env["S3_BUCKET_DATA"] == "stock-intel-data-prod"
    assert env["S3_PREFIX_DATA"] == "data"
    assert env["AWS_ACCESS_KEY_ID"] == "${{ secrets.AWS_ACCESS_KEY_ID }}"
    assert env["AWS_SECRET_ACCESS_KEY"] == "${{ secrets.AWS_SECRET_ACCESS_KEY }}"
    assert env["AWS_DEFAULT_REGION"] == "us-east-1"


def test_scheduled_guard_skip_is_reported_as_skip(monkeypatch, capsys):
    sunday = datetime(2026, 9, 20, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setattr(runner, "now_ny", lambda: sunday)
    monkeypatch.setattr(runner, "run_registry_guarded", lambda **kwargs: False)

    runner.main()

    output = capsys.readouterr().out
    assert "[SKIP] options universe build was not executed." in output
    assert "[OK] options universe build completed." not in output


def test_scheduled_execution_is_reported_as_success(monkeypatch, capsys):
    sunday = datetime(2026, 9, 20, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setattr(runner, "now_ny", lambda: sunday)
    monkeypatch.setattr(runner, "run_registry_guarded", lambda **kwargs: True)

    runner.main()

    assert "[OK] options universe build completed." in capsys.readouterr().out


def test_non_sunday_skip_uses_defined_date_and_does_not_call_guard(monkeypatch, capsys):
    monday = datetime(2026, 9, 21, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setattr(runner, "now_ny", lambda: monday)
    monkeypatch.setattr(
        runner,
        "run_registry_guarded",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("guard called")),
    )

    runner.main()

    assert "Today (2026-09-21) is not Sunday" in capsys.readouterr().out


def test_manual_dispatch_preserves_registry_bypass(monkeypatch):
    observed = {}
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")

    def guarded(**kwargs):
        observed.update(kwargs)
        return True

    monkeypatch.setattr(runner, "run_registry_guarded", guarded)

    runner.main()

    assert observed["job_name"] == runner.JOB_NAME
    assert observed["fn"] is runner.run_profile
    assert observed["bypass_registry"] is True
