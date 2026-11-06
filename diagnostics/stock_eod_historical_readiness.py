"""Offline analysis of retained Stock EOD evidence; no application/network imports."""
from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re

import pandas as pd
import pyarrow.parquet as pq

NY = 'America/New_York'
START, END = date(2026, 10, 12), date(2026, 11, 4)
MAX_BYTES = 128 * 1024 * 1024
MAX_RUNS = 100


def timestamp(value):
    if value is None or pd.isna(value):
        return None
    t = pd.Timestamp(value)
    if pd.isna(t):
        return None
    if t.tzinfo is None:
        raise ValueError('Evidence timestamps must include an explicit timezone')
    return t.tz_convert('UTC')


def rendered(t):
    return {'utc': t.tz_convert('UTC').isoformat(), 'eastern': t.tz_convert(NY).isoformat()} if t is not None else None


def elapsed(a, b):
    if a is None or b is None:
        return None
    if b < a:
        raise ValueError('Evidence timestamp order is reversed')
    return (b - a).total_seconds()


def number(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def session_state(row, intended):
    session = row.get('terminal_session')
    evidence = row.get('session_evidence')
    if session is None or pd.isna(session) or not evidence or pd.isna(evidence):
        return 'UNVERIFIED'
    observed = date.fromisoformat(str(session))
    if observed < intended:
        return 'STALE_SESSION'
    if observed > intended:
        return 'FUTURE_SESSION'
    return 'MATCHING_SESSION_NOT_FINALITY_PROOF'


def analyze(index_path: Path):
    """Read a local catalog and return reports in memory; never write inputs."""
    index_path = index_path.resolve()
    if index_path.stat().st_size > 1024 * 1024:
        raise ValueError('Catalog exceeds 1 MiB')
    catalog = json.loads(index_path.read_text())
    entries = catalog['runs']
    if not 0 < len(entries) <= MAX_RUNS:
        raise ValueError('Expected 1..100 runs')
    consumed = 0
    inventory, runtime, observations = [{'path': str(index_path), 'size_bytes': index_path.stat().st_size, 'sha256': sha256(index_path.read_bytes()).hexdigest(), 'source': 'catalog'}], [], []
    seen_runs, seen_families = set(), set()

    def read(spec):
        nonlocal consumed
        path = (index_path.parent / spec['path']).resolve()
        if not path.is_relative_to(index_path.parent):
            raise ValueError('Evidence must stay within the catalog directory')
        size = path.stat().st_size
        consumed += size
        if consumed > MAX_BYTES:
            raise ValueError('Evidence exceeds 128 MiB total')
        data = path.read_bytes()
        digest = sha256(data).hexdigest()
        if size != spec['size_bytes'] or digest != spec['sha256']:
            raise ValueError(f'Evidence integrity mismatch: {path.name}')
        inventory.append({'path': str(path), 'size_bytes': size, 'sha256': digest,
                          'source': spec.get('source', 'UNKNOWN')})
        return data

    for entry in entries:
        run = entry['workflow_run']
        run_id = run['id']
        if run_id in seen_runs:
            raise ValueError('Duplicate workflow run')
        seen_runs.add(run_id)
        if run['name'] != 'Stocks Daily EOD' or run['status'] != 'completed':
            raise ValueError('Only completed Stocks Daily EOD runs are supported')
        jobs = [j for j in entry['jobs'] if j['name'] == 'daily-eod-stocks']
        if len(jobs) != 1:
            raise ValueError('Supply exactly one latest-attempt daily-eod-stocks job')
        job = jobs[0]
        if job['run_id'] != run_id or job['status'] != 'completed':
            raise ValueError('Job/run identity or completion mismatch')
        intended = date.fromisoformat(entry['intended_session'])
        if not START <= intended <= END:
            raise ValueError('Intended session outside 2026-10-12..2026-11-04')
        created, started = timestamp(run.get('created_at')), timestamp(run.get('run_started_at'))
        js, je = timestamp(job.get('started_at')), timestamp(job.get('completed_at'))
        scheduled = timestamp(entry.get('scheduled_at'))
        logs = read(entry['log']).decode() if entry.get('log') else ''
        skipped = bool(re.search(r'\[GUARD\] skip stocks_eod:', logs))
        marked = '[GUARD] marked success stocks_eod in execution registry' in logs
        family_ids = set(re.findall(r'prod-\d{8}T\d{6}Z-[a-f0-9]{8}', logs))
        # A log can mention previous families: bind family start only to the manifest ID.
        rt = {'run_id': run_id, 'intended_session': str(intended), 'event': run['event'],
              'commit': run.get('head_sha'), 'conclusion': run.get('conclusion'),
              'scheduled_at': rendered(scheduled), 'created_at': rendered(created),
              'workflow_started_at': rendered(started), 'job_started_at': rendered(js),
              'job_completed_at': rendered(je), 'job_seconds': elapsed(js, je),
              'schedule_to_created_seconds': elapsed(scheduled, created),
              'created_to_job_seconds': elapsed(created, js),
              'workflow_to_job_seconds': elapsed(started, js),
              'registry_skipped': skipped if logs else None,
              'registry_success_mark_verified': marked,
              'registry_skip_reason': next((line.split('[GUARD] skip stocks_eod:', 1)[1].strip() for line in logs.splitlines() if '[GUARD] skip stocks_eod:' in line), None),
              'concurrency_delay_seconds': None,
              'publication_at': None, 'family_started_at': None,
              'first_weekly_started_at': None, 'last_weekly_completed_at': None,
              'weekly_envelope_seconds': None, 'weekly_timing_complete': False,
              'completion_by_1930': None, 'counterfactual_1700_completion': None}
        steps = [s for s in job.get('steps', []) if s['name'] == 'Run stocks EOD (DST-aware guarded)']
        rt['profile_seconds'] = elapsed(timestamp(steps[0].get('started_at')),
                                         timestamp(steps[0].get('completed_at'))) if len(steps) == 1 else None
        if entry.get('manifest'):
            if skipped:
                raise ValueError('Skipped invocation cannot own a manifest')
            spec = entry['manifest']
            fid = spec['family_run_id']
            if fid in seen_families or not re.fullmatch(r'prod-\d{8}T\d{6}Z-[a-f0-9]{8}', fid):
                raise ValueError('Invalid or duplicate family identity')
            seen_families.add(fid)
            if logs and fid not in family_ids:
                raise ValueError('Manifest family not corroborated by supplied log')
            data = read(spec)
            parquet = pq.ParquetFile(BytesIO(data))
            if parquet.metadata.num_rows > 30000 or len(parquet.schema_arrow) > 96:
                raise ValueError('Manifest row/column cap exceeded')
            frame = parquet.read().to_pandas()
            required = {'record_type', 'family_run_id', 'symbol', 'interval'}
            if not required <= set(frame):
                raise ValueError('Missing manifest identity columns')
            if set(frame.family_run_id.dropna()) != {fid}:
                raise ValueError('Manifest family identity mismatch')
            native = frame.loc[frame.record_type == 'interval'].copy()
            if native.duplicated(['symbol', 'interval']).any():
                raise ValueError('Duplicate native observation')
            if not set(native.interval) <= {'1d', '1wk', '1mo'}:
                raise ValueError('Unexpected native interval')
            family_start = timestamp(pd.Timestamp(fid.split('-')[1]))
            rt['family_started_at'] = rendered(family_start)
            for line in logs.splitlines():
                if '[FAMILY_PUBLISH][COMMIT_OK]' in line and f'family_run_id={fid}' in line:
                    rt['publication_at'] = rendered(timestamp(line.split()[0]))
            starts, ends = [], []
            complete = True
            for _, row in native.iterrows():
                fs, fe = timestamp(row.get('fetch_started_at')), timestamp(row.get('fetch_completed_at'))
                elapsed(fs, fe)
                state = session_state(row, intended)
                record = {'run_id': run_id, 'family_run_id': fid, 'intended_session': str(intended),
                          'symbol': row.symbol, 'interval': row.interval,
                          'fetch_started_at': fs.isoformat() if fs is not None else None,
                          'fetch_completed_at': fe.isoformat() if fe is not None else None,
                          'session_state': state, 'finality': 'UNPROVEN', 'partial_period_status': 'UNKNOWN: no completeness flag in health schema'}
                for field in ['terminal_session', 'session_evidence', 'provider_version', 'adjustment_mode',
                              'session_semantics', 'acquisition_status', 'quality_status', 'quality_reason',
                              'acquisition_error', 'finalization_status', 'processing_status', 'provider_metadata_json']:
                    value = row.get(field)
                    record[field] = None if value is None or pd.isna(value) else str(value)
                for field in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                    record['normalized_' + field] = number(row.get('raw_' + field))
                    record[field + '_recording'] = 'NOT_RECORDED' if 'raw_' + field not in row.index else ('FINITE' if record['normalized_' + field] is not None else 'MISSING_OR_NONFINITE')
                observations.append(record)
                if row.interval == '1wk':
                    complete &= fs is not None and fe is not None
                    if fs is not None: starts.append(fs)
                    if fe is not None: ends.append(fe)
            rt['weekly_observations'] = len(native.loc[native.interval == '1wk'])
            rt['weekly_timing_complete'] = bool(complete and starts and ends)
            first, last = min(starts) if starts else None, max(ends) if ends else None
            rt['first_weekly_started_at'], rt['last_weekly_completed_at'] = rendered(first), rendered(last)
            rt['weekly_envelope_seconds'] = elapsed(first, last)
            if rt['weekly_timing_complete']:
                target = pd.Timestamp(f'{intended} 19:30', tz=NY)
                rt['completion_by_1930'] = bool(last <= target)
                if js is not None:
                    phase = elapsed(js, last)
                    counter = pd.Timestamp(f'{intended} 17:00', tz=NY) + pd.Timedelta(seconds=phase)
                    rt['counterfactual_1700_completion'] = rendered(counter)
                    rt['counterfactual_assumption'] = '17:00 actual job start; same observed duration; excludes schedule/queue delay'
        runtime.append(rt)

    revisions = []
    groups = {}
    for row in observations:
        groups.setdefault((row['symbol'], row['interval'], row['intended_session']), []).append(row)
    for key, rows in groups.items():
        timed = sorted([r for r in rows if r['fetch_completed_at']], key=lambda r: r['fetch_completed_at'])
        for before, after in zip(timed, timed[1:]):
            pair = {'symbol': key[0], 'interval': key[1], 'intended_session': key[2],
                    'before_family': before['family_run_id'], 'after_family': after['family_run_id'],
                    'comparison': 'MATCHING_SESSION' if all(r['session_state'] == 'MATCHING_SESSION_NOT_FINALITY_PROOF' for r in [before, after]) else 'UNVERIFIED_SESSION_COMPARISON',
                    'period_identity': 'DAILY_SESSION' if key[1] == '1d' else 'UNKNOWN: interval labels not in health schema'}
            for field in ['close', 'volume']:
                a, b = before['normalized_' + field], after['normalized_' + field]
                pair[field + '_before'], pair[field + '_after'] = a, b
                pair[field + '_delta'] = b - a if a is not None and b is not None else None
                pair[field + '_transition'] = before[field + '_recording'] + '_TO_' + after[field + '_recording']
            revisions.append(pair)
    for rt in runtime:
        now = rt['job_started_at']
        predecessors = [p for p in runtime if p['registry_success_mark_verified'] and now and p['job_completed_at'] and
                        timestamp(p['job_completed_at']['utc']) <= timestamp(now['utc']) and p['run_id'] != rt['run_id']]
        predecessor = max(predecessors, key=lambda p: p['job_completed_at']['utc']) if predecessors else None
        rt['preceding_verified_success_run'] = predecessor['run_id'] if predecessor else None
        rt['preceding_success_scope'] = 'supplied catalog only; registry row/history not reconstructed'
    def distribution(values):
        values = sorted(v for v in values if v is not None)
        return {'count': len(values), 'min_seconds': min(values) if values else None,
                'median_seconds': float(pd.Series(values, dtype=float).median()) if values else None,
                'max_seconds': max(values) if values else None}

    quality = []
    for family in sorted({r['family_run_id'] for r in observations}):
        for interval in ['1d', '1wk', '1mo']:
            rows = [r for r in observations if r['family_run_id'] == family and r['interval'] == interval]
            quality.append({'family_run_id': family, 'interval': interval, 'recorded_count': len(rows),
                            'missing_normalized_close': sum(r['close_recording'] == 'MISSING_OR_NONFINITE' for r in rows),
                            'unrecorded_close': sum(r['close_recording'] == 'NOT_RECORDED' for r in rows),
                            'missing_normalized_adj_close': sum(r['adj_close_recording'] == 'MISSING_OR_NONFINITE' for r in rows),
                            'invalid_normalized_ohlcv': sum(any(r['normalized_' + f] is None for f in ['open','high','low','close','volume']) or (r['normalized_volume'] is not None and r['normalized_volume'] < 0) for r in rows),
                            'accepted_processing': sum(r['processing_status'] == 'accepted' for r in rows),
                            'unverified_sessions': sum(r['session_state'] == 'UNVERIFIED' for r in rows)})
    summary = {'decision': 'NO_GO: historical evidence does not prove provider finality at 17:00',
               'runs': len(runtime), 'observations': len(observations),
               'weekly_runs_with_complete_recorded_timing': sum(r['weekly_timing_complete'] for r in runtime),
               'recorded_weekly_envelopes_completed_by_1930': sum(r['completion_by_1930'] is True for r in runtime),
               'runtime_distributions': {status: distribution([r['profile_seconds'] for r in runtime if r['conclusion'] == status and r['registry_skipped'] is False]) for status in ['success', 'failure']},
               'weekly_envelope_distribution': distribution([r['weekly_envelope_seconds'] for r in runtime if r['weekly_timing_complete']]),
               'session_states': {str(k): int(v) for k, v in pd.Series([r['session_state'] for r in observations], dtype=str).value_counts().items()},
               'operational_feasibility': 'UNPROVEN: requires complete universe, 17:00 start evidence and queue-tail margin',
               'required_corrections': ['DST-aware local eligibility and late-start policy', 'Session-aware registry/cooldown treatment of late recovery', 'Preserve serialization and 90 percent coverage safeguard'],
               'minimum_live_diagnostic': 'Separately authorized nonpublishing 17:00/17:30/18:00 observations with later and next-session controls; session-calendar verification and Close/Volume revision analysis. No live calls are implemented.',
               'limitations': ['Native timestamps bound loader calls, not HTTP request/retry timestamps.',
                              'Missing scheduled_at is UNKNOWN; no cron slot is inferred.',
                              'Created-to-job delay is not attributed to concurrency without independent evidence.',
                              'Finite normalized OHLCV, matching sessions and stable observations do not prove finality.',
                              'Health schema omits terminal interval labels, full payloads and explicit volume finality.',
                              'Catalog completeness and source authenticity require external verification; hashes establish byte identity only.',
                              'Recorded weekly timing completeness does not establish routed-universe completeness.']}
    return {'inventory': inventory, 'runtime': runtime, 'observations': observations, 'revisions': revisions, 'field_quality': quality, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output, root = args.output_dir.resolve(), args.catalog.resolve().parent
    production = Path(__file__).resolve().parents[1] / 'data'
    if output == root or root in output.parents or output == production or production in output.parents:
        parser.error('Output must be outside evidence directory and repository data/')
    reports = analyze(args.catalog)
    output.mkdir(parents=True, exist_ok=False)
    for name, report in reports.items():
        (output / (name + '.json')).write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n')


if __name__ == '__main__':
    main()
