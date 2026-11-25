import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from core import notify
from core.guard import run_registry_guarded

ROOT = Path(__file__).resolve().parents[1]


def runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / '.github/scripts' / f'run_{name}_intraday_4h_guarded.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(notify, 'DATA', tmp_path)
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test')
    return tmp_path


def test_transport_redacts_and_classifies(monkeypatch):
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'secret')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'secret')
    def timeout(*args, **kwargs):
        raise requests.ReadTimeout('secret')
    monkeypatch.setattr(notify.requests, 'post', timeout)
    with pytest.raises(notify.TelegramDeliveryError, match='ReadTimeout') as err:
        notify.send_telegram_message('private')
    assert 'secret' not in str(err.value)
    monkeypatch.setattr(notify.requests, 'post', lambda *a, **kw: SimpleNamespace(ok=False, status_code=502, text='secret'))
    with pytest.raises(notify.TelegramDeliveryError) as err:
        notify.send_telegram_message('private')
    assert err.value.status == 502 and 'secret' not in str(err.value)


def test_unconfirmed_attempt_is_not_retried(local_storage, monkeypatch, capsys):
    calls = []
    def failed(*a, **kw):
        calls.append(1)
        raise notify.TelegramDeliveryError('ReadTimeout')
    monkeypatch.setattr(notify, 'send_telegram_message', failed)
    args = dict(job='stocks_intraday_4h', kind='repair_telemetry', identity='private')
    with pytest.raises(notify.TelegramDeliveryError):
        notify.send_4h_tracked('private', **args)
    assert not notify.send_4h_tracked('private', **args)
    assert len(calls) == 1
    assert 'private' not in capsys.readouterr().out


def test_attempt_store_failure_prevents_send(local_storage, monkeypatch, capsys):
    monkeypatch.setattr(notify.storage, 'exists', lambda p: (_ for _ in ()).throw(OSError('secret')))
    monkeypatch.setattr(notify, 'send_telegram_message', lambda *a, **kw: pytest.fail('sent'))
    assert not notify.send_4h_tracked('x', job='stocks', kind='repair', identity='x')
    assert 'NOT_ATTEMPTED' in capsys.readouterr().out


def test_combo_fingerprints_and_unconfirmed_delivery(local_storage, monkeypatch):
    path = local_storage / 'combo_demo.parquet'
    def combo(symbol):
        pd.DataFrame([{'symbol': symbol, 'signal': 'long'}]).to_parquet(path)
    combo('AAA')
    calls = []
    def send(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise notify.TelegramDeliveryError('HTTPError', 502)
    monkeypatch.setattr(notify, 'send_telegram_message', send)
    call = lambda: notify.notify_combo_signals('demo', only_if_changed=True, tracked_4h_job='futures_intraday_4h')
    with pytest.raises(notify.TelegramDeliveryError):
        call()
    assert notify.load_alert_fingerprint('demo') is None
    call()
    assert len(calls) == 1
    combo('BBB')
    call()
    confirmed = notify.load_alert_fingerprint('demo')
    assert confirmed and 'BBB' in confirmed
    call()
    assert len(calls) == 2
    combo('CCC')
    call()
    assert len(calls) == 3


def test_success_followed_by_fingerprint_failure_propagates(local_storage, monkeypatch):
    pd.DataFrame([{'symbol': 'AAA', 'signal': 'long'}]).to_parquet(local_storage / 'combo_demo.parquet')
    calls = []
    monkeypatch.setattr(notify, 'send_telegram_message', lambda *a, **kw: calls.append(1))
    monkeypatch.setattr(notify, 'save_alert_fingerprint', lambda *a, **kw: (_ for _ in ()).throw(OSError('fingerprint')))
    with pytest.raises(OSError, match='fingerprint'):
        notify.notify_combo_signals('demo', only_if_changed=True, tracked_4h_job='stocks_intraday_4h')
    notify.notify_combo_signals('demo', only_if_changed=True, tracked_4h_job='stocks_intraday_4h')
    assert len(calls) == 1


@pytest.mark.parametrize('name,kind,status', [('stocks', 'combo_signal', None), ('futures', 'combo_signal', 502)])
def test_delivery_failure_still_marks_registry(name, kind, status, monkeypatch, capsys):
    mod = runner(name)
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(mod.storage, 'exists', lambda p: False) if name == 'stocks' else None
    monkeypatch.setattr(mod, 'run_combo_health', lambda **kw: [])
    monkeypatch.setattr(mod, 'print_results', lambda results: None)
    def fail(*a, **kw):
        raise notify.TelegramDeliveryError('HTTPError' if status else 'ReadTimeout', status)
    monkeypatch.setattr(mod, 'notify_combo_signals', fail)
    marks = []
    monkeypatch.setattr('core.guard.mark_registry_execution', lambda **kw: marks.append(kw))
    run_registry_guarded(job_name=mod.JOB_NAME, fn=mod.run_profile, bypass_registry=True)
    assert len(marks) == 1
    output = capsys.readouterr().out
    assert 'DELIVERY_UNCONFIRMED' in output and 'SUCCEEDED' in output


def test_repair_timeout_does_not_block_health_or_signal(monkeypatch):
    mod = runner('stocks')
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(mod.storage, 'exists', lambda p: True)
    monkeypatch.setattr(mod.storage, 'load_parquet', lambda p: pd.DataFrame([{'symbol': 'A', 'source': 'x', 'bars_total': 10, 'bars_repaired': 5}]))
    def fail(*a, **kw):
        raise notify.TelegramDeliveryError('ReadTimeout')
    monkeypatch.setattr(mod, 'send_4h_tracked', fail)
    seen = []
    monkeypatch.setattr(mod, 'run_combo_health', lambda **kw: seen.append('health') or [])
    monkeypatch.setattr(mod, 'print_results', lambda x: None)
    monkeypatch.setattr(mod, 'notify_combo_signals', lambda *a, **kw: seen.append('signal'))
    mod.run_profile()
    assert seen == ['health', 'signal']


@pytest.mark.parametrize('stage', ['ingestion', 'combo', 'health', 'registry', 'signal'])
def test_core_failures_propagate(stage, monkeypatch):
    mod = runner('futures')
    count = []
    def subprocess_run(*a, **kw):
        count.append(1)
        if stage == 'ingestion' or (stage == 'combo' and len(count) == 3):
            raise RuntimeError(stage)
    monkeypatch.setattr(mod.subprocess, 'run', subprocess_run)
    def health(**kw):
        if stage == 'health':
            raise RuntimeError(stage)
        return []
    monkeypatch.setattr(mod, 'run_combo_health', health)
    monkeypatch.setattr(mod, 'print_results', lambda x: None)
    def signal(*a, **kw):
        if stage == 'signal':
            raise ValueError(stage)
    monkeypatch.setattr(mod, 'notify_combo_signals', signal)
    def mark(**kw):
        if stage == 'registry':
            raise RuntimeError(stage)
    monkeypatch.setattr('core.guard.mark_registry_execution', mark)
    with pytest.raises((RuntimeError, ValueError), match=stage):
        run_registry_guarded(job_name=mod.JOB_NAME, fn=mod.run_profile, bypass_registry=True)
