import json

import pandas as pd
import pytest

from core import health, notify


def coverage(counts, expected=100):
    symbols = [f"S{i}" for i in range(expected)]
    snapshots = {tf: pd.DataFrame({"symbol": symbols[:count]})
                 for tf, count in zip(("daily", "weekly", "monthly"), counts)}
    return health.stock_eod_final_coverage("synthetic", snapshots, symbols)


@pytest.mark.parametrize("counts,expected,failing", [
    ((100, 100, 100), 100, []),
    ((90, 90, 90), 100, []),
    ((89, 100, 100), 100, ["daily"]),
    ((100, 89, 100), 100, ["weekly"]),
    ((100, 100, 89), 100, ["monthly"]),
    ((89, 0, 88), 100, ["daily", "weekly", "monthly"]),
    ((2293, 2364, 9), 2364, ["monthly"]),
    ((2298, 1613, 2330), 2330, ["weekly"]),
    ((2239, 2364, 2364), 2364, []),
    ((2331, 2364, 2364), 2364, []),
])
def test_final_coverage_floor_and_historical_shapes(counts, expected, failing):
    result = coverage(counts, expected)
    assert result["expected_count"] == expected
    assert result["failing_timeframes"] == failing
    assert result["passed"] == (not failing)
    assert result["threshold_percent"] == health.STOCK_EOD_CATASTROPHIC_COVERAGE_PERCENT
    if failing:
        with pytest.raises(RuntimeError, match="catastrophic coverage"):
            health.require_stock_eod_final_coverage(result)
    else:
        health.require_stock_eod_final_coverage(result)


def test_denominator_and_numerator_are_normalized_distinct_current_memberships():
    snapshots = {tf: pd.DataFrame({"symbol": [" a ", "A", "B", "OLD", None]})
                 for tf in ("daily", "weekly", "monthly")}
    result = health.stock_eod_final_coverage("new", snapshots, ["A", " b ", "a"])
    assert result["expected_count"] == 2
    assert result["final_counts"] == dict.fromkeys(snapshots, 2)
    assert result["passed"]
    # The very same stored rows do not imply complete coverage of a larger run.
    result = health.stock_eod_final_coverage("larger", snapshots, ["A", "B", "C"])
    assert result["expected_count"] == 3 and not result["passed"]


def test_empty_expected_universe_cannot_pass():
    with pytest.raises(ValueError, match="nonempty"):
        health.stock_eod_final_coverage("empty", {}, [])


def test_shortlist_and_options_health_contracts_remain_independent(monkeypatch):
    assert coverage((99, 100, 100))["passed"]
    combo = pd.DataFrame({"symbol": ["OTHER"]})
    monkeypatch.setattr(health.storage, "exists", lambda path: True)
    monkeypatch.setattr(health.storage, "load_parquet", lambda path: combo)
    monkeypatch.setattr(health, "load_universe_symbols", lambda path: {"REQUIRED"})
    short = health.run_combo_health(["shortlist"], universe_csv="shortlist_stocks.csv")
    assert any(not result.ok and "REQUIRED" in result.details for result in short)
    opts = health.run_combo_health(["options"], universe_csv=None)
    assert len(opts) == 1 and opts[0].ok  # not complete options membership
    combo.drop(combo.index, inplace=True)
    assert not health.run_combo_health(["options"], universe_csv=None)[0].ok


def test_failure_notification_uses_existing_route_and_plain_text(monkeypatch):
    report = coverage((98, 69, 100))
    report["shortlist_missing"] = ["REQUIRED"]
    calls = []
    monkeypatch.setattr(notify, "telegram_chat_id_for_route", lambda route: "existing-" + route)
    monkeypatch.setattr(notify, "send_telegram_message", lambda text, **kwargs: calls.append((text, kwargs)))
    notify.notify_stock_eod_failure("product_failure_with_[punctuation]", report, ["shortlist: missing REQUIRED"])
    assert len(calls) == 1
    text, kwargs = calls[0]
    for value in ["synthetic", "D: 98 / 100", "W: 69 / 100", "M: 100 / 100", "floor=90%", "weekly", "REQUIRED"]:
        assert value in text
    assert kwargs == {"chat_id": "existing-stocks_eod", "parse_mode": None}


@pytest.mark.parametrize("plain", [True, False])
def test_plain_failure_transport_does_not_change_default_signal_markdown(monkeypatch, plain):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    sent = []
    from types import SimpleNamespace
    monkeypatch.setattr(notify.requests, "post", lambda url, **kw: sent.append(kw["data"]) or SimpleNamespace(ok=True))
    kwargs = {"parse_mode": None} if plain else {}
    notify.send_telegram_message("text", chat_id="test", **kwargs)
    assert ("parse_mode" not in sent[0]) if plain else sent[0]["parse_mode"] == "Markdown"


def test_standalone_cli_enforces_floor_after_run_without_removing_results(monkeypatch, tmp_path):
    from jobs import run_stock_eod_family as runner
    marker = tmp_path / "published"
    def run_family(*, coverage_report):
        marker.write_text("published and mirrored")
        coverage_report.write_text(json.dumps(coverage((100, 50, 100))))
        return tmp_path / "health.parquet"
    monkeypatch.setattr(runner, "run_family", run_family)
    monkeypatch.setattr(runner.sys, "argv", ["run_stock_eod_family.py"])
    with pytest.raises(RuntimeError, match="weekly"):
        runner.main()
    assert marker.read_text() == "published and mirrored"
