from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/intraday_4h.yml"
ONE_HOUR_WORKFLOW = ROOT / ".github/workflows/futures_intraday_1h.yml"
OBSOLETE = (
    "futures_intraday_orchestrator.yml",
    "futures_intraday_4h.yml",
    "stocks_intraday_4h.yml",
)
CRON = "1,21,41 * * * 0-5"


def load(path):
    return yaml.safe_load(path.read_text())


def triggers(workflow):
    return workflow[True]  # PyYAML 1.1 parses the unquoted `on` key as True.


def test_shared_workflow_topology_and_manual_selector():
    workflow = load(WORKFLOW)
    assert triggers(workflow)["schedule"] == [{"cron": CRON}]
    assert "concurrency" not in workflow
    assert set(workflow["jobs"]) == {"futures_4h", "stocks_4h"}

    selector = triggers(workflow)["workflow_dispatch"]["inputs"]["profile"]
    assert selector["required"] is True
    assert selector["type"] == "choice"
    assert selector["default"] == "futures_4h"
    assert selector["options"] == ["futures_4h", "stocks_4h", "all_4h"]

    assert "futures_4h" in workflow["jobs"]["futures_4h"]["if"]
    assert "stocks_4h" not in workflow["jobs"]["futures_4h"]["if"]
    assert "stocks_4h" in workflow["jobs"]["stocks_4h"]["if"]
    assert "futures_4h" not in workflow["jobs"]["stocks_4h"]["if"]
    assert "all_4h" in workflow["jobs"]["futures_4h"]["if"]
    assert "all_4h" in workflow["jobs"]["stocks_4h"]["if"]


def test_jobs_are_independent_and_have_complete_environment():
    workflow = load(WORKFLOW)
    expected = {
        "DATA_BACKEND": "s3",
        "S3_BUCKET_DATA": "stock-intel-data-prod",
        "S3_PREFIX_DATA": "data",
        "AWS_ACCESS_KEY_ID": "${{ secrets.AWS_ACCESS_KEY_ID }}",
        "AWS_SECRET_ACCESS_KEY": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
        "AWS_DEFAULT_REGION": "us-east-1",
        "TELEGRAM_BOT_TOKEN": "${{ secrets.TELEGRAM_BOT_TOKEN }}",
        "TELEGRAM_CHAT_ID": "${{ secrets.TELEGRAM_CHAT_ID }}",
    }
    contracts = {
        "futures_4h": ("futures-intraday", "run_futures_intraday_4h_guarded.py"),
        "stocks_4h": ("stocks-intraday-4h", "run_stocks_intraday_4h_guarded.py"),
    }
    for name, (group, runner) in contracts.items():
        job = workflow["jobs"][name]
        assert "needs" not in job
        assert "strategy" not in job
        assert job["concurrency"] == {"group": group, "cancel-in-progress": False}
        assert job["env"] == expected
        commands = "\n".join(str(step.get("run", "")) for step in job["steps"])
        assert runner in commands
        assert sum(other in commands for _, other in contracts.values()) == 1
        assert "registry" not in commands.lower()


def test_futures_1h_schedule_and_cross_workflow_serialization():
    shared = load(WORKFLOW)
    hourly = load(ONE_HOUR_WORKFLOW)
    assert triggers(hourly)["schedule"] == [{"cron": CRON}]
    assert hourly["concurrency"] == {
        "group": "futures-intraday",
        "cancel-in-progress": False,
    }
    assert shared["jobs"]["futures_4h"]["concurrency"] == hourly["concurrency"]


def test_obsolete_workflows_are_absent():
    workflows = ROOT / ".github/workflows"
    for name in OBSOLETE:
        assert not (workflows / name).exists()
