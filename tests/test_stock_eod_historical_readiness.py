"""Synthetic offline evidence only; never invoke the diagnostic CLI or production."""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from diagnostics.stock_eod_historical_readiness import analyze, timestamp


def fixture(tmp_path, *, session='2026-11-02', start='2026-11-02T22:00:00Z', end='2026-11-03T00:25:00Z'):
    fid = 'prod-20261102T220000Z-01234567'
    rows = [dict(record_type='interval', family_run_id=fid, symbol='A', interval=i,
                 fetch_started_at=start, fetch_completed_at=end, terminal_session=session,
                 session_evidence='native_daily_interval_label' if i == '1d' else None,
                 raw_open=100., raw_high=110., raw_low=90., raw_close=105., raw_adj_close=104.,
                 raw_volume=1000., processing_status='accepted') for i in ['1d', '1wk', '1mo']]
    p = tmp_path / 'manifest.parquet'; pd.DataFrame(rows).to_parquet(p)
    def spec(path):
        b = path.read_bytes()
        return dict(path=path.name, size_bytes=len(b), sha256=sha256(b).hexdigest())
    log = tmp_path / 'job.log'
    log.write_text(f'2026-11-03T01:00:00Z [FAMILY_PUBLISH][COMMIT_OK] family_run_id={fid}\n[GUARD] marked success stocks_eod in execution registry\n')
    run = dict(id=1, name='Stocks Daily EOD', status='completed', event='workflow_dispatch',
               conclusion='success', created_at='2026-11-02T21:59:00Z', run_started_at='2026-11-02T21:59:01Z')
    job = dict(name='daily-eod-stocks', run_id=1, status='completed', started_at='2026-11-02T22:00:00Z',
               completed_at='2026-11-03T01:30:00Z', steps=[dict(name='Run stocks EOD (DST-aware guarded)',
               started_at='2026-11-02T22:00:01Z', completed_at='2026-11-03T01:29:00Z')])
    catalog = dict(runs=[dict(workflow_run=run, jobs=[job], intended_session='2026-11-02',
                             log=spec(log), manifest=dict(**spec(p), family_run_id=fid))])
    index = tmp_path / 'catalog.json'
    def save(): index.write_text(json.dumps(catalog))
    save()
    return index, catalog, save, rows, spec


def test_request_times_not_family_or_publication_and_no_false_go(tmp_path):
    index, _, _, _, _ = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    out = analyze(index); rt = out['runtime'][0]
    assert rt['first_weekly_started_at']['utc'] == '2026-11-02T22:00:00+00:00'
    assert rt['last_weekly_completed_at']['eastern'] == '2026-11-02T19:25:00-05:00'
    assert rt['weekly_envelope_seconds'] == 8700
    assert rt['publication_at']['utc'] == '2026-11-03T01:00:00+00:00'
    assert rt['counterfactual_1700_completion']['utc'] == '2026-11-03T00:25:00+00:00'
    assert rt['completion_by_1930']
    assert out['summary']['decision'].startswith('NO_GO')
    assert out['observations'][0]['finality'] == 'UNPROVEN'
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert out['field_quality'][1]['unverified_sessions'] == 1


@pytest.mark.parametrize('value', ['2026-11-02T22:00:00', 'bad timestamp'])
def test_ambiguous_timestamps_rejected(value):
    with pytest.raises(ValueError): timestamp(value)


def test_dst_conversion():
    assert timestamp('2026-10-12T21:00:00Z').tz_convert('America/New_York').hour == 17
    assert timestamp('2026-11-02T22:00:00Z').tz_convert('America/New_York').hour == 17


@pytest.mark.parametrize('column', ['fetch_started_at', 'fetch_completed_at'])
def test_missing_request_time_never_substitutes(tmp_path, column):
    index, cat, save, rows, spec = fixture(tmp_path)
    for row in rows: row[column] = None
    p = tmp_path / 'manifest.parquet'; pd.DataFrame(rows).to_parquet(p)
    cat['runs'][0]['manifest'].update(spec(p));save()
    rt = analyze(index)['runtime'][0]
    assert not rt['weekly_timing_complete']
    assert rt['completion_by_1930'] is None
    assert rt['counterfactual_1700_completion'] is None


def test_integrity_failure(tmp_path):
    index, _, _, _, _ = fixture(tmp_path)
    with (tmp_path / 'manifest.parquet').open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='integrity'): analyze(index)


def test_duplicates_and_wrong_family_rejected(tmp_path):
    index, cat, save, rows, spec = fixture(tmp_path)
    p = tmp_path / 'manifest.parquet';pd.DataFrame(rows + [rows[0]]).to_parquet(p)
    cat['runs'][0]['manifest'].update(spec(p));save()
    with pytest.raises(ValueError, match='Duplicate native'): analyze(index)
    rows[0]['family_run_id'] = 'wrong'
    pd.DataFrame(rows).to_parquet(p);cat['runs'][0]['manifest'].update(spec(p));save()
    with pytest.raises(ValueError, match='identity mismatch'): analyze(index)


def test_stale_session_not_finite_readiness(tmp_path):
    index, _, _, _, _ = fixture(tmp_path, session='2026-10-30')
    out=analyze(index)
    assert out['observations'][0]['session_state'] == 'STALE_SESSION'
    assert out['observations'][1]['session_state'] == 'UNVERIFIED'


def test_skip_has_no_acquisition_and_no_runtime_success_sample(tmp_path):
    index, cat, save, _, spec = fixture(tmp_path)
    entry=cat['runs'][0];entry.pop('manifest')
    log=tmp_path/'job.log';log.write_text('[GUARD] skip stocks_eod: last execution 2.00h ago < check_window_hours=20\n')
    entry['log']=spec(log);save()
    out=analyze(index)
    assert out['runtime'][0]['registry_skipped']
    assert out['runtime'][0]['first_weekly_started_at'] is None
    assert out['summary']['runtime_distributions']['success']['count'] == 0


def test_revision_keeps_unverified_pairs_distinct(tmp_path):
    index, cat, save, rows, spec = fixture(tmp_path)
    second=json.loads(json.dumps(cat['runs'][0]));second['workflow_run']['id']=2;second['jobs'][0]['run_id']=2
    second.pop('log');fid='prod-20261103T010000Z-abcdef01'
    for row in rows:
        row.update(family_run_id=fid, fetch_started_at='2026-11-03T01:00:00Z',fetch_completed_at='2026-11-03T01:01:00Z',raw_close=106.,raw_volume=1100.)
    p=tmp_path/'second.parquet';pd.DataFrame(rows).to_parquet(p)
    second['manifest']=dict(**spec(p), family_run_id=fid)
    cat['runs'].append(second);save()
    out=analyze(index)
    assert len(out['revisions'])==3
    assert out['revisions'][0]['close_delta']==1
    assert out['revisions'][0]['volume_delta']==100
    assert out['revisions'][1]['comparison']=='UNVERIFIED_SESSION_COMPARISON'


def test_missing_schedule_and_queue_cause_remain_unknown(tmp_path):
    index, _, _, _, _=fixture(tmp_path)
    rt=analyze(index)['runtime'][0]
    assert rt['scheduled_at'] is None and rt['schedule_to_created_seconds'] is None
    assert rt['created_to_job_seconds']==60
    assert rt['concurrency_delay_seconds'] is None


def test_future_session_and_unrecorded_fields(tmp_path):
    index, cat, save, rows, spec=fixture(tmp_path, session='2026-11-03')
    for row in rows: row.pop('raw_close')
    p=tmp_path/'manifest.parquet';pd.DataFrame(rows).to_parquet(p)
    cat['runs'][0]['manifest'].update(spec(p));save()
    out=analyze(index)
    assert out['observations'][0]['session_state']=='FUTURE_SESSION'
    assert out['observations'][0]['close_recording']=='NOT_RECORDED'
    assert out['field_quality'][0]['unrecorded_close']==1


def test_evidence_escape_and_out_of_scope_session(tmp_path):
    index, cat, save, _, _=fixture(tmp_path)
    cat['runs'][0]['intended_session']='2026-11-05';save()
    with pytest.raises(ValueError, match='outside'):analyze(index)
    cat['runs'][0]['intended_session']='2026-11-02'
    cat['runs'][0]['log']['path']='../outside.log';save()
    with pytest.raises(ValueError, match='within'):analyze(index)


def test_recorded_null_to_finite_revision(tmp_path):
    index, cat, save, rows, spec=fixture(tmp_path)
    second=json.loads(json.dumps(cat['runs'][0]));second['workflow_run']['id']=2;second['jobs'][0]['run_id']=2
    second.pop('log');fid='prod-20261103T010000Z-abcdef02'
    for row in rows:
        row.update(family_run_id=fid,fetch_completed_at='2026-11-03T01:01:00Z',raw_close=None)
    p=tmp_path/'later.parquet';pd.DataFrame(rows).to_parquet(p)
    second['manifest']=dict(**spec(p),family_run_id=fid);cat['runs'].append(second);save()
    pair=analyze(index)['revisions'][0]
    assert pair['close_delta'] is None
    assert pair['close_transition']=='FINITE_TO_MISSING_OR_NONFINITE'
