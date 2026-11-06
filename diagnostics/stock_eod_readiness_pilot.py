"""Standalone, manually authorized native-observation pilot; never publishes data."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from hashlib import sha256
from io import BytesIO
import json
import math
from numbers import Integral
from pathlib import Path
import re
import sys
import time
import numpy as np

import pandas as pd
import pyarrow.parquet as pq

NY = 'America/New_York'
FAILED = 'prod-20261103T001713Z-044f683b'
BASE = ('SPY', 'QQQ', 'IWM', 'GLD', 'XLF', 'AAPL', 'MSFT', 'NVDA')
INTERVALS = ('1d', '1wk', '1mo')
MAX_SYMBOLS, MAX_HISTORY, MAX_HTTP = 14, 42, 60
QUOTES = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
META = ('exchangeTimezoneName', 'instrumentType', 'priceHint', 'regularMarketTime',
        'regularMarketPrice', 'currentTradingPeriod', 'lastTrade')


def finite(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def atom(value):
    if value is None: return {'kind': 'None'}
    if value is pd.NA: return {'kind': 'pd.NA'}
    if value is pd.NaT: return {'kind': 'NaT'}
    if isinstance(value, (bool, np.bool_)): return {'kind': 'bool', 'value': bool(value)}
    if isinstance(value, (pd.Timestamp, date)): return {'kind': 'timestamp', 'value': value.isoformat()}
    if isinstance(value, str):
        if re.search(r'(?i)([a-z][a-z0-9+.-]*://|www\.|bearer\s|(?:token|cookie|password|crumb|authorization)\s*[:=])', value):
            return {'kind': 'redacted'}
        return {'kind': 'string', 'value': value[:256], 'truncated': len(value)>256}
    if isinstance(value, Integral):
        return {'kind': 'integer', 'value': int(value)} if int(value).bit_length()<=128 else {'kind':'oversized_integer'}
    if isinstance(value, (float, np.floating)):
        n=float(value)
        if math.isfinite(n): return {'kind': 'finite', 'value': n, 'hex': n.hex()}
        return {'kind':'nonfinite','value':'NaN' if math.isnan(n) else ('+infinity' if n>0 else '-infinity')}
    return {'kind': 'unsupported', 'type': type(value).__name__}


def metadata(value, depth=0):
    if depth > 4: return {'kind': 'depth_limit'}
    if isinstance(value, dict):
        items=[]
        # Stable order without repr/str on arbitrary keys or values.
        keys=sorted(k for k in value if isinstance(k,str))
        for key in keys[:24]:
            sensitive=bool(re.search(r'(?i)(credential|cookie|header|url|token|crumb|password|secret|authorization|api.?key)',key))
            items.append([atom(key), {'kind':'redacted'} if sensitive else metadata(value[key],depth+1)])
        return {'kind':'dict','items':items,'truncated':len(value)>24 or len(keys)!=len(value)}
    if isinstance(value,(list,tuple)):
        return {'kind':'list' if isinstance(value,list) else 'tuple',
                'items':[metadata(v,depth+1) for v in value[:24]],'truncated':len(value)>24}
    return atom(value)


def frame_evidence(frame):
    if len(frame) > 256 or len(frame.columns) > 16 or isinstance(frame.columns, pd.MultiIndex):
        raise ValueError('Native frame exceeds supported bounded schema')
    return {'columns': [str(c) for c in frame.columns], 'dtypes': [str(d) for d in frame.dtypes],
            'index_name': str(frame.index.name), 'index_timezone': str(getattr(frame.index, 'tz', None)),
            'rows': [{'label': atom(label), 'values': [atom(v) for v in row]} for label, row in zip(frame.index, frame.itertuples(index=False, name=None))]}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def panel_from_manifest(path, expected_hash):
    if not re.fullmatch('[a-f0-9]{64}', expected_hash): raise ValueError('Verified SHA-256 is required')
    if path.stat().st_size > 16 * 1024 * 1024: raise ValueError('Manifest exceeds 16 MiB')
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected_hash: raise ValueError('Manifest hash mismatch')
    parquet = pq.ParquetFile(BytesIO(data))
    if parquet.metadata.num_rows > 10000 or len(parquet.schema_arrow) > 96: raise ValueError('Manifest bounds exceeded')
    frame = parquet.read().to_pandas()
    required = {'family_run_id','record_type','interval','symbol','fetch_started_at','raw_close','raw_adj_close','raw_open','raw_high','raw_low','raw_volume'}
    if not required <= set(frame): raise ValueError('Missing cohort evidence columns')
    if set(frame.family_run_id.dropna()) != {FAILED}: raise ValueError('Wrong incident family')
    weekly = frame.loc[(frame.record_type == 'interval') & (frame.interval == '1wk')].copy()
    if len(weekly) != 2335 or weekly.symbol.duplicated().any(): raise ValueError('Incomplete weekly cohort')
    for value in weekly.fetch_started_at:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None: raise ValueError('Missing or naive cohort timestamp')
    times = pd.to_datetime(weekly.fetch_started_at, utc=True, errors='coerce')
    if times.isna().any(): raise ValueError('Missing cohort timestamps')
    post = times >= pd.Timestamp('2026-11-03T01:00:00Z')
    closes = weekly.raw_close.map(finite).notna()
    adjusted = weekly.raw_adj_close.map(finite).notna()
    ohlv = weekly[['raw_open','raw_high','raw_low','raw_volume']].apply(lambda col: col.map(finite)).notna().all(axis=1)
    if not ohlv.all() or (weekly.raw_volume < 0).any(): raise ValueError('Unexpected incident OHLV signature')
    if int((~post).sum()) != 686 or not (closes & adjusted).loc[~post].all() or int((post & closes & adjusted).sum()) != 13 or int((post & ~closes & ~adjusted).sum()) != 1636:
        raise ValueError('Cohort evidence does not reproduce verified 686/13/1636 partition')
    if not set(BASE) <= set(weekly.symbol): raise ValueError('Missing base controls')
    roles = {s: ['liquid_equity_or_etf_control'] for s in BASE}
    for role, subset in [('affected', weekly.loc[post & ~closes]), ('post_boundary_exception', weekly.loc[post & closes & adjusted])]:
        subset = subset.sort_values(['raw_volume','symbol'])
        for position, rank in [(0,'low'), (len(subset)//2,'middle'), (len(subset)-1,'high')]:
            symbol = subset.iloc[position].symbol
            if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,9}', symbol): raise ValueError('Unsupported panel symbol')
            roles.setdefault(symbol, []).append(role + '_' + rank + '_incident_share_volume')
    if len(roles) > MAX_SYMBOLS: raise ValueError('Panel budget exceeded')
    return [{'symbol': s, 'roles': roles[s]} for s in sorted(roles)]


def plan(session, stage, calendar):
    target = pd.Timestamp(session)
    if not calendar.is_session(target): raise ValueError('Target is not an XNYS session')
    close = calendar.session_close(target).tz_convert(NY)
    next_day = date.fromisoformat(session) + timedelta(days=1)
    wall = {'primary': f'{session} 17:00', 'later': f'{session} 18:00', 'reference': f'{next_day} 08:00'}[stage]
    start = pd.Timestamp(wall, tz=NY)
    if start <= close: raise ValueError('Sampling must follow target session close')
    week_end = target + pd.Timedelta(days=4-target.weekday())
    month_end = target + pd.offsets.MonthEnd(0)
    return {'session': session, 'stage': stage, 'window_start': start,
            'latest_start': start + pd.Timedelta(minutes=5), 'deadline': start + pd.Timedelta(minutes=15),
            'request_start': pd.Timestamp(f'{target.date()-timedelta(days=90)} 00:00', tz=NY),
            'request_end_cap': pd.Timestamp(f'{next_day} 00:00', tz=NY),
            'session_close': close,
            'week_state': 'COMPLETED_WEEK' if calendar.sessions_in_range(target, week_end)[-1].date() == target.date() else 'WEEK_TO_DATE',
            'month_state': 'COMPLETED_MONTH' if calendar.sessions_in_range(target, month_end)[-1].date() == target.date() else 'MONTH_TO_DATE'}


def strict_assess(frame, meta, interval, p):
    if frame.empty: return {'status': 'EMPTY', 'session_evidence': None}
    label = pd.Timestamp(frame.index[-1])
    if label.tzinfo is None or meta.get('exchangeTimezoneName') != NY:
        return {'status': 'AMBIGUOUS_TIMEZONE', 'session_evidence': None}
    label = label.tz_convert(NY)
    target = date.fromisoformat(p['session'])
    floor = target if interval == '1d' else (target - timedelta(days=target.weekday()) if interval == '1wk' else target.replace(day=1))
    if not floor <= label.date() <= target:
        return {'status': 'WRONG_PERIOD_OR_NEXT_SESSION', 'session_evidence': None}
    row = frame.iloc[-1]
    values = {q: finite(row.get(q)) for q in QUOTES}
    required = ['Open','High','Low','Close','Volume']
    valid = all(values[q] is not None for q in required) and values['Volume'] >= 0
    if valid:
        valid = values['Low'] <= values['High'] and values['Low'] <= values['Close'] <= values['High']
    evidence = 'native_daily_label' if interval == '1d' else None
    if interval != '1d':
        trade = meta.get('lastTrade', {})
        if not isinstance(trade,dict): trade={}
        try:
            time = pd.Timestamp(trade.get('Time'))
            price = finite(trade.get('Price'))
            if time.tzinfo is not None and time.tz_convert(NY).date() == target and price is not None and values['Close'] is not None and abs(price-values['Close']) <= 1e-4:
                evidence = 'price_bound_yfinance_lastTrade'
        except (TypeError, ValueError): pass
    return {'status': 'CURRENT_SESSION_OBSERVED' if valid and evidence else ('INVALID_OHLCV' if not valid else 'UNVERIFIED_SESSION'),
            'adj_close_observed': values['Adj Close'] is not None,
            'session_evidence': evidence, 'terminal_label': label.isoformat(), 'values': values,
            'period_state': 'COMPLETED_SESSION' if interval == '1d' else p['week_state' if interval == '1wk' else 'month_state'],
            'finality': 'NOT_PROVEN_BY_THIS_OBSERVATION'}


def assess(frame, meta, interval, p):
    """Keep field validity independent from the strict session-attribution gate."""
    row = frame.iloc[-1] if not frame.empty else None
    fields = {q: ('ABSENT' if q not in frame.columns else
                  'EMPTY' if row is None else
                  'FINITE' if finite(row.get(q)) is not None else 'MISSING_OR_NONFINITE') for q in QUOTES}
    values = {q: finite(row.get(q)) if row is not None else None for q in QUOTES}
    required = ('Open','High','Low','Close','Volume')
    complete = all(fields[q] == 'FINITE' for q in required)
    valid = complete and values['Volume'] >= 0 and values['Low'] <= values['High'] and values['Low'] <= values['Close'] <= values['High']
    result = strict_assess(frame, meta, interval, p)
    result.update(field_completeness=fields, required_fields_complete=complete,
                  numerical_validity='VALID_OHLCV' if valid else ('EMPTY' if frame.empty else 'INVALID_OHLCV'))
    # These remain observed facts even when the strict attribution gate exits early.
    result.setdefault('values', values)
    result.setdefault('terminal_label', pd.Timestamp(frame.index[-1]).isoformat() if not frame.empty else None)
    return result


def collect(panel, p, history, clock, output, transport_count, versions, incident_hash):
    now = clock().tz_convert(NY)
    if not p['window_start'] <= now <= p['latest_start']:
        output.mkdir(parents=True, exist_ok=False)
        missed={'schema_version':3,'plan':{k:v.isoformat() if isinstance(v,pd.Timestamp) else v for k,v in p.items()},'collection_complete':False,'decision':'MISSED_OBSERVATION_WINDOW','observed_at':now.isoformat(),'attempted_history_calls':0,'http_calls':0}
        (output/'observations.json').write_text(json.dumps(missed,indent=2)+'\n')
        return missed
    if not 1 <= len(panel) <= MAX_SYMBOLS or len({x['symbol'] for x in panel}) != len(panel): raise ValueError('Invalid symbol budget')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 3,
              'production_parity':False,
              'acquisition_profile':{'keepna':True,'redirects':False,'transport_retries':0,'metadata':'public_getter_transport_locked','reference_end':'target_midnight_exclusive'}, 'incident_family_run_id': FAILED, 'incident_sha256': incident_hash, 'panel': panel,
              'plan': {k: v.isoformat() if isinstance(v, pd.Timestamp) else v for k,v in p.items()},
              'versions': versions, 'adjustment_mode': 'raw_unadjusted', 'prepost': False,
              'attempted_history_calls': 0, 'history_budget': MAX_HISTORY, 'http_budget': MAX_HTTP, 'observations': []}
    try:
        for item in panel:
            for interval in INTERVALS:
                if len(report['observations']) >= MAX_HISTORY or clock() >= p['deadline']: raise ValueError('Pilot request/deadline budget exhausted')
                started = clock()
                end = min(started, p['request_end_cap'])
                report['attempted_history_calls'] += 1
                frame, meta = history(item['symbol'], interval, p['request_start'], end)
                completed = clock()
                native = frame_evidence(frame)
                if len(json.dumps(native, allow_nan=False).encode()) > 128*1024: raise ValueError('Native frame evidence exceeds 128 KiB')
                observation = {'symbol': item['symbol'], 'interval': interval,
                               'started_at': started.isoformat(), 'completed_at': completed.isoformat(),
                               'requested_start': p['request_start'].isoformat(), 'requested_end': end.isoformat(),
                               'native_frame': native, 'native_payload_sha256': digest(native),
                               'metadata': {k: metadata(meta[k]) for k in META if k in meta},
                               'assessment': assess(frame, meta, interval, p)}
                observation['metadata_sha256'] = digest(observation['metadata'])
                if len(json.dumps(observation['metadata'], allow_nan=False).encode()) > 32*1024: raise ValueError('Metadata evidence exceeds 32 KiB')
                report['observations'].append(observation)
                if completed > p['deadline']: raise ValueError('Pilot completed after deadline')
        report['collection_complete'] = True
    except Exception as exc:
        # Evidence may be partial. Do not export exception text, URLs or credentials.
        report['collection_complete'] = False
        report['failure_type'] = type(exc).__name__
    report['http_calls'] = transport_count()
    report['provider'] = 'Yahoo via yfinance; native return frames, not HTTP response bodies'
    report['decision'] = 'CANDIDATE_FOR_COMPARISON' if report['collection_complete'] and all(o['assessment']['status'] == 'CURRENT_SESSION_OBSERVED' for o in report['observations']) else 'NO_GO_OR_UNVERIFIED'
    (output/'observations.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def decode(value):
    if isinstance(value, dict) and 'kind' not in value:
        return {k: decode(v) for k, v in value.items()}
    kind = value['kind']
    if kind == 'dict': return {decode(k):decode(v) for k,v in value['items'] if decode(k) is not None}
    if kind in {'list','tuple'}: return [decode(v) for v in value['items']]
    if kind in {'finite', 'string', 'integer','bool'}: return value['value']
    if kind == 'timestamp': return pd.Timestamp(value['value'])
    return None


def reassess(observation, p):
    native = observation['native_frame']
    rows = native['rows']
    frame = pd.DataFrame([[decode(v) for v in r['values']] for r in rows],
                         columns=native['columns'], index=pd.DatetimeIndex([decode(r['label']) for r in rows]))
    return assess(frame, decode(observation['metadata']), observation['interval'], p)


def compare(reports):
    if any(r.get('schema_version') != 3 for r in reports): raise ValueError('Require corrected schema version 3')
    if [r['plan']['stage'] for r in reports] != ['primary','later','reference']: raise ValueError('Require ordered primary/later/reference evidence')
    key = lambda r: (r['plan']['session'], r['incident_sha256'], r['panel'], r['versions'], r.get('acquisition_profile'))
    if not all(key(r) == key(reports[0]) for r in reports): raise ValueError('Pilot identity/panel/version mismatch')
    maps = []
    expected_keys = {(item['symbol'], interval) for item in reports[0]['panel'] for interval in INTERVALS}
    if not 1 <= len(reports[0]['panel']) <= MAX_SYMBOLS or len(expected_keys) != len(reports[0]['panel'])*3: raise ValueError('Invalid comparison panel')
    for r in reports:
        if not r['collection_complete'] or len(r['observations']) != len(r['panel'])*3: raise ValueError('Incomplete pilot evidence')
        if r['attempted_history_calls'] > MAX_HISTORY or r['http_calls'] > MAX_HTTP: raise ValueError('Reported request budget exceeded')
        target = date.fromisoformat(r['plan']['session'])
        next_day = target + timedelta(days=1)
        wall = {'primary':f'{target} 17:00', 'later':f'{target} 18:00', 'reference':f'{next_day} 08:00'}[r['plan']['stage']]
        expected_window = pd.Timestamp(wall,tz=NY)
        cap = pd.Timestamp(f'{next_day} 00:00',tz=NY)
        if pd.Timestamp(r['plan']['window_start']) != expected_window or pd.Timestamp(r['plan']['deadline']) != expected_window+pd.Timedelta(minutes=15) or pd.Timestamp(r['plan']['request_end_cap']) != cap: raise ValueError('Changed phase plan')
        mapping = {}
        for o in r['observations']:
            k = (o['symbol'],o['interval'])
            if k in mapping or digest(o['native_frame']) != o['native_payload_sha256']: raise ValueError('Duplicate or changed native evidence')
            if digest(o['metadata']) != o.get('metadata_sha256'): raise ValueError('Changed metadata evidence')
            if reassess(o, r['plan']) != o['assessment']: raise ValueError('Assessment does not match preserved evidence')
            started, completed = pd.Timestamp(o['started_at']), pd.Timestamp(o['completed_at'])
            window, deadline = pd.Timestamp(r['plan']['window_start']), pd.Timestamp(r['plan']['deadline'])
            if not window <= started <= completed <= deadline or pd.Timestamp(o['requested_end']) > pd.Timestamp(r['plan']['request_end_cap']): raise ValueError('Observation timing outside approved plan')
            mapping[k] = o
        if set(mapping) != expected_keys: raise ValueError('Unexpected symbol/interval set')
        if min(pd.Timestamp(o['started_at']) for o in mapping.values()) > expected_window+pd.Timedelta(minutes=5): raise ValueError('Late phase start')
        maps.append(mapping)
    if not all(set(m) == set(maps[0]) for m in maps): raise ValueError('Observation key mismatch')
    result = []
    for k in maps[0]:
        obs = [m[k] for m in maps]
        assessments = [o['assessment'] for o in obs]
        same = len({a['terminal_label'] for a in assessments}) == 1
        labels = [a['terminal_label'] for a in assessments]
        period_candidate = all(a['status'] in {'CURRENT_SESSION_OBSERVED','UNVERIFIED_SESSION','INVALID_OHLCV'} for a in assessments)
        attributed = all(a['status'] == 'CURRENT_SESSION_OBSERVED' for a in assessments)
        numerical = all(a['numerical_validity'] == 'VALID_OHLCV' for a in assessments)
        comparable = period_candidate and same and all(label is not None for label in labels)
        deltas = {field: [a['values'][field]-assessments[0]['values'][field]
                         if comparable and a['values'][field] is not None and assessments[0]['values'][field] is not None else None
                         for a in assessments[1:]] for field in QUOTES}
        adj_eligible = comparable and all(a['values']['Adj Close'] is not None for a in assessments)
        core_stable = comparable and numerical and all(v==0 for field,ds in deltas.items() if field!='Adj Close' for v in ds)
        if not comparable:
            status='NOT_COMPARABLE_PERIOD_OR_LABEL'
        elif not numerical:
            status='NUMERICALLY_INVALID_OBSERVATIONS'
        elif not attributed:
            status='UNVERIFIED_PERIOD_STABLE' if core_stable and all(v==0 for field,ds in deltas.items() if field!='Adj Close' or adj_eligible for v in ds) else 'UNVERIFIED_PERIOD_NUMERICAL_DIFFERENCES'
        else:
            status='STABLE_OBSERVED_NOT_FINALITY_PROOF' if all(v==0 for ds in deltas.values() for v in ds) else ('CORE_STABLE_ADJ_CLOSE_UNAVAILABLE' if core_stable and not adj_eligible else 'REVISED_OR_PERIOD_AMBIGUOUS')
        result.append({'symbol':k[0],'interval':k[1],'terminal_labels':labels,
                       'same_terminal_label':same,'deltas_vs_primary':deltas,
                       'native_payload_changed_vs_primary':[o['native_payload_sha256'] != obs[0]['native_payload_sha256'] for o in obs[1:]],
                       'field_completeness':[a['field_completeness'] for a in assessments],
                       'numerical_validity':[a['numerical_validity'] for a in assessments],
                       'session_attribution_statuses':[a['status'] for a in assessments],
                       'comparison_scope':'SESSION_ATTRIBUTED_OBSERVATIONS_NOT_FINALITY' if attributed else 'UNVERIFIED_PERIOD_STABILITY_NOT_SAME_SESSION_REVISIONS',
                       'core_comparison_eligible':comparable and numerical,'adj_close_comparison_eligible':adj_eligible,
                       'production_readiness':'NO_GO_FINALITY_NOT_PROVEN' if attributed else 'NO_GO_SESSION_ATTRIBUTION_UNVERIFIED',
                       'status':status})
    return {'decision':'REVIEW_REQUIRED_NO_AUTOMATIC_PRODUCTION_GO','comparisons':result}


def bounded_session_class(base, clock, deadline):
    class BoundedSession(base):
        calls = 0
        metadata_only = False
        def request(self, method, url, **kwargs):
            if self.metadata_only: raise RuntimeError('Metadata-only access cannot acquire')
            if self.calls >= MAX_HTTP or clock() >= deadline: raise RuntimeError('Transport budget exhausted')
            self.calls += 1
            kwargs.update(allow_redirects=False,max_redirects=0,timeout=min(10,max(0.1,(deadline-clock()).total_seconds())))
            response = super().request(method,url,**kwargs)
            if 300 <= response.status_code < 400: raise RuntimeError('Redirect denied by pilot budget policy')
            return response
    return BoundedSession


def observed_metadata(ticker, session):
    session.metadata_only = True
    try:
        view=ticker.get_history_metadata(repair=False)
        return {k:view[k] for k in META if k in view}
    except Exception:
        return {}
    finally:
        session.metadata_only = False


def arm(p, clock, sleep):
    remaining=(p['window_start']-clock()).total_seconds()
    if remaining > 1800: raise ValueError('Arm no more than 30 minutes before phase')
    while clock() < p['window_start']:
        sleep(min(1, (p['window_start']-clock()).total_seconds()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture = sub.add_parser('capture')
    for flag in ['manifest','manifest-sha256','session','output-dir']: capture.add_argument('--'+flag, required=True)
    capture.add_argument('--stage', required=True, choices=['primary','later','reference'])
    analysis = sub.add_parser('compare')
    for flag in ['primary','later','reference']: analysis.add_argument('--'+flag, required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'compare':
        reports=[]
        for path in [args.primary,args.later,args.reference]:
            if path.stat().st_size > 16*1024*1024: raise ValueError('Pilot artifact exceeds 16 MiB')
            reports.append(json.loads(path.read_text()))
        print(json.dumps(compare(reports), indent=2, allow_nan=False));return
    # Dependencies imported only for explicitly requested live capture.
    import yfinance as yf
    import curl_cffi
    from curl_cffi.requests import Session
    import exchange_calendars as xc
    if yf.__version__ != '1.7.0' or curl_cffi.__version__ != '0.16.3': raise ValueError('Unsupported transport/provider version')
    clock = lambda: pd.Timestamp.now(tz='UTC')
    p = plan(args.session,args.stage,xc.get_calendar('XNYS'))
    panel = panel_from_manifest(Path(args.manifest),args.manifest_sha256)
    output = Path(args.output_dir).resolve()
    production = Path(__file__).resolve().parents[1]/'data'
    if output == production or production in output.parents: raise ValueError('Output cannot be production data')
    BoundedSession = bounded_session_class(Session, clock, p['deadline'])
    # Arm up to 30 minutes early so setup/download latency precedes the sample.
    arm(p, clock, time.sleep)
    session = BoundedSession(impersonate='chrome',retry=0)
    def history(symbol,interval,start,end):
        ticker = yf.Ticker(symbol,session=session)
        frame = ticker.history(interval=interval,start=start,end=end,auto_adjust=False,back_adjust=False,
                               actions=False,prepost=False,repair=False,keepna=True,timeout=10)
        meta = observed_metadata(ticker,session)
        return frame, meta
    try:
        report = collect(panel,p,history,clock,output,lambda:session.calls,
                         {'yfinance':yf.__version__,'curl_cffi':curl_cffi.__version__,'pandas':pd.__version__,'exchange_calendars':xc.__version__},args.manifest_sha256)
    finally:
        session.close()
    if report['decision'] != 'CANDIDATE_FOR_COMPARISON': sys.exit(2)


if __name__ == '__main__': main()
