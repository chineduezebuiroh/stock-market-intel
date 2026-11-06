"""Offline fixtures only. No real provider or workflow execution."""
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from diagnostics import stock_eod_readiness_pilot as d


class Calendar:
    def is_session(self, t): return t.weekday() < 5
    def session_close(self, t): return pd.Timestamp(str(t.date())+' 16:00',tz=d.NY)
    def sessions_in_range(self, a, b): return pd.date_range(a,b,freq='B')


def frame(session='2026-11-02', close=105.):
    return pd.DataFrame({'Open':[100.], 'High':[110.], 'Low':[90.], 'Close':[close],
                         'Adj Close':[104.], 'Volume':[1000.]},index=pd.DatetimeIndex([session],tz=d.NY))


def meta():
    return {'exchangeTimezoneName':d.NY,'lastTrade':{'Time':pd.Timestamp('2026-11-02 16:00',tz=d.NY),'Price':105.}}


def test_cohorts_are_evidence_derived_and_bounded(tmp_path):
    names=list(d.BASE)+[f'S{i:04}' for i in range(2327)]
    rows=[]
    for i,s in enumerate(names):
        post=i>=686;bad=i>=699
        rows.append(dict(family_run_id=d.FAILED,record_type='interval',interval='1wk',symbol=s,
                         fetch_started_at='2026-11-03T01:00:04Z' if post else '2026-11-03T00:59:00Z',
                         raw_close=None if bad else 105.,raw_adj_close=None if bad else 104.,
                         raw_open=100.,raw_high=110.,raw_low=90.,raw_volume=float(i+1)))
    p=tmp_path/'manifest.parquet';pd.DataFrame(rows).to_parquet(p);h=sha256(p.read_bytes()).hexdigest()
    panel=d.panel_from_manifest(p,h)
    assert len(panel)<=14 and set(d.BASE)<=set(x['symbol'] for x in panel)
    assert sum(any(r.startswith('affected') for r in x['roles']) for x in panel)==3
    assert sum(any(r.startswith('post_boundary_exception') for r in x['roles']) for x in panel)==3
    with pytest.raises(ValueError,match='hash'):d.panel_from_manifest(p,'a'*64)
    rows[-1]['raw_close']=105.;pd.DataFrame(rows).to_parquet(p)
    with pytest.raises(ValueError,match='partition'):d.panel_from_manifest(p,sha256(p.read_bytes()).hexdigest())


@pytest.mark.parametrize('session,offset', [('2026-10-12','-04:00'),('2026-11-02','-05:00')])
def test_local_windows_dst(session,offset):
    p=d.plan(session,'primary',Calendar())
    assert p['window_start'].isoformat().endswith('17:00:00'+offset)
    assert p['week_state']=='WEEK_TO_DATE'
    assert p['month_state']=='MONTH_TO_DATE'


def test_friday_and_month_end_and_non_session():
    p=d.plan('2026-10-30','reference',Calendar())
    assert p['week_state']=='COMPLETED_WEEK' and p['month_state']=='COMPLETED_MONTH'
    assert p['request_end_cap']==pd.Timestamp('2026-10-31',tz=d.NY)
    with pytest.raises(ValueError,match='session'): d.plan('2026-11-01','primary',Calendar())


def test_next_session_bar_and_unverified_weekly_fail_closed():
    p=d.plan('2026-11-02','reference',Calendar())
    assert d.assess(frame('2026-11-03'),meta(),'1d',p)['status']=='WRONG_PERIOD_OR_NEXT_SESSION'
    assert d.assess(frame(),{'exchangeTimezoneName':d.NY},'1wk',p)['status']=='UNVERIFIED_SESSION'
    assert d.assess(frame(close=float('nan')),meta(),'1wk',p)['status']=='INVALID_OHLCV'
    assert d.assess(frame(),meta(),'1wk',p)['finality']=='NOT_PROVEN_BY_THIS_OBSERVATION'


def make_report(tmp_path,stage='primary'):
    p=d.plan('2026-11-02',stage,Calendar());now=p['window_start'].tz_convert('UTC')
    original=frame();before=original.copy(deep=True);calls=[]
    def history(symbol,interval,start,end):
        calls.append((symbol,interval,end))
        return original,meta()
    report=d.collect([{'symbol':'SPY','roles':['control']}],p,history,lambda:now,tmp_path/stage,lambda:3,{'yfinance':'1.7.0'},'a'*64)
    pd.testing.assert_frame_equal(original,before)
    assert len(calls)==3
    assert all(end<=p['request_end_cap'] for _,_,end in calls)
    return report


def test_capture_preserves_full_native_data_and_caps_reference_end(tmp_path):
    report=make_report(tmp_path,'reference')
    assert report['collection_complete'] and report['decision']=='CANDIDATE_FOR_COMPARISON'
    native=report['observations'][0]['native_frame']
    assert native['columns']==list(d.QUOTES)
    assert d.digest(native)==report['observations'][0]['native_payload_sha256']


def test_no_requests_outside_window_or_over_symbol_budget(tmp_path):
    p=d.plan('2026-11-02','primary',Calendar())
    def forbidden(*a):pytest.fail('provider called')
    missed=d.collect([],p,forbidden,lambda:p['window_start']+pd.Timedelta(minutes=6),tmp_path/'missed',lambda:0,{},'a'*64)
    assert missed['decision']=='MISSED_OBSERVATION_WINDOW' and missed['http_calls']==0
    with pytest.raises(ValueError,match='symbol'):
        d.collect([{'symbol':str(i)} for i in range(15)],p,forbidden,lambda:p['window_start'],tmp_path/'x',lambda:0,{},'a'*64)


def test_failure_retains_partial_artifact_without_exception_text(tmp_path):
    p=d.plan('2026-11-02','primary',Calendar())
    def bad(*a):raise RuntimeError('secret URL')
    r=d.collect([{'symbol':'SPY'}],p,bad,lambda:p['window_start'],tmp_path/'x',lambda:1,{},'a'*64)
    assert not r['collection_complete'] and r['decision']=='NO_GO_OR_UNVERIFIED'
    assert 'secret' not in (tmp_path/'x'/'observations.json').read_text()


def test_comparison_stability_revision_and_ambiguous_identity(tmp_path):
    reports=[make_report(tmp_path,s) for s in ['primary','later','reference']]
    out=d.compare(reports)
    assert all(c['status']=='STABLE_OBSERVED_NOT_FINALITY_PROOF' for c in out['comparisons'])
    changed=deepcopy(reports);o=changed[2]['observations'][0];o['native_frame']['rows'][0]['values'][5]=d.atom(1001.);o['native_payload_sha256']=d.digest(o['native_frame']);o['assessment']['values']['Volume']=1001.
    assert d.compare(changed)['comparisons'][0]['status']=='REVISED_OR_PERIOD_AMBIGUOUS'
    changed=deepcopy(reports);changed[2]['plan']['session']='2026-11-03'
    with pytest.raises(ValueError,match='identity'):d.compare(changed)
    changed=deepcopy(reports);changed[0]['observations'][0]['native_payload_sha256']='wrong'
    with pytest.raises(ValueError,match='changed'):d.compare(changed)


def test_workflow_manual_only_no_production_credentials_or_mutations():
    raw=(Path(__file__).parents[1]/'.github/workflows/stock-eod-readiness-pilot.yml').read_text()
    c=yaml.safe_load(raw)
    assert set(c.get('on',c.get(True)))=={'workflow_dispatch'}
    assert c['permissions']=={'contents':'read','actions':'read'}
    assert not any(s in raw for s in ['AWS_ACCESS_KEY_ID','secrets.','aws s3','run_stock_eod_family','schedule:'])
    job=c['jobs']['sample-native-observations']
    assert job['timeout-minutes']==55
    assert job['steps'][-1]['with']['retention-days']==30
    assert job['steps'][-1]['if']=='${{ always() }}'


def test_transport_budget_and_redirects_without_network():
    now=pd.Timestamp('2026-11-02T22:00:00Z')
    class Base:
        def request(self, method, url, **kwargs):
            assert kwargs['allow_redirects'] is False and kwargs['max_redirects']==0
            assert kwargs['timeout']<=10
            return type('Response', (), {'status_code':200})()
    cls=d.bounded_session_class(Base,lambda:now,now+pd.Timedelta(minutes=15))
    transport=cls()
    for _ in range(60):transport.request('GET','https://example.invalid')
    with pytest.raises(RuntimeError,match='budget'):transport.request('GET','https://example.invalid')
    assert transport.calls==60
    class Redirect(Base):
        def request(self,*a,**k):return type('Response', (), {'status_code':302})()
    with pytest.raises(RuntimeError,match='Redirect'):
        d.bounded_session_class(Redirect,lambda:now,now+pd.Timedelta(minutes=15))().request('GET','https://example.invalid')


def test_comparison_does_not_trust_altered_assessment(tmp_path):
    reports=[make_report(tmp_path,s) for s in ['primary','later','reference']]
    reports[0]['observations'][0]['assessment']['values']['Close']=999.
    with pytest.raises(ValueError,match='Assessment'):d.compare(reports)


def test_full_panel_has_exact_42_history_calls(tmp_path):
    p=d.plan('2026-11-02','primary',Calendar());calls=[]
    def history(*args):calls.append(args);return frame(),meta()
    panel=[{'symbol':f'S{i}','roles':['synthetic']} for i in range(14)]
    r=d.collect(panel,p,history,lambda:p['window_start'],tmp_path/'full',lambda:42,{},'a'*64)
    assert r['collection_complete'] and r['attempted_history_calls']==42 and len(calls)==42


def test_late_completion_fails_with_retained_evidence(tmp_path):
    p=d.plan('2026-11-02','primary',Calendar());now=[p['window_start']]
    def history(*args):now[0]=p['deadline']+pd.Timedelta(seconds=1);return frame(),meta()
    r=d.collect([{'symbol':'SPY'}],p,history,lambda:now[0],tmp_path/'late',lambda:1,{},'a'*64)
    assert not r['collection_complete'] and len(r['observations'])==1


def test_half_day_uses_calendar_close():
    class HalfDay(Calendar):
        def session_close(self,t):return pd.Timestamp(str(t.date())+' 13:00',tz=d.NY)
    assert d.plan('2026-11-27','primary',HalfDay())['session_close'].hour==13


def test_native_integer_and_nonfinite_distinctions():
    assert d.atom(2**60+1)['value']==2**60+1
    assert d.atom(float('nan'))['value']=='NaN'
    assert d.atom(float('inf'))['value']=='+infinity'
    assert d.atom(float('-inf'))['value']=='-infinity'


def test_comparison_rejects_shifted_phase_and_request_budget(tmp_path):
    reports=[make_report(tmp_path,s) for s in ['primary','later','reference']]
    changed=deepcopy(reports);changed[2]['plan']['deadline']='2026-11-03T10:00:00-05:00'
    with pytest.raises(ValueError,match='phase plan'):d.compare(changed)
    changed=deepcopy(reports);changed[0]['http_calls']=61
    with pytest.raises(ValueError,match='budget'):d.compare(changed)


def test_governed_open_outside_range_and_optional_adjusted_close(tmp_path):
    p=d.plan('2026-11-02','primary',Calendar());f=frame();f.loc[:,'Open']=200.
    assert d.assess(f,meta(),'1d',p)['status']=='CURRENT_SESSION_OBSERVED'
    f.loc[:,'Adj Close']=float('nan')
    a=d.assess(f,meta(),'1d',p)
    assert a['status']=='CURRENT_SESSION_OBSERVED' and not a['adj_close_observed']
    f.loc[:,'Close']=111.
    assert d.assess(f,meta(),'1d',p)['status']=='INVALID_OHLCV'


def test_metadata_containers_types_redaction_and_determinism():
    value={'nested':[True,False,None,pd.NA,pd.NaT,pd.Timestamp('2026-11-02',tz=d.NY),float('inf'),float('-inf'),float('nan'),{'normal':'text','cookies':'SECRET','url':'https://secret.invalid'}], 'headers':{'Authorization':'SECRET'}}
    encoded=d.metadata(value)
    assert json.dumps(encoded,allow_nan=False)==json.dumps(d.metadata(value),allow_nan=False)
    assert 'SECRET' not in json.dumps(encoded) and 'https://' not in json.dumps(encoded)
    decoded=d.decode(encoded)
    assert decoded['nested'][0] is True and decoded['nested'][1] is False
    assert d.atom(True)['kind']=='bool'
    assert d.metadata(object())['kind']=='unsupported'
    assert d.metadata(list(range(30)))['truncated']


def test_early_arm_and_missed_window(tmp_path):
    p=d.plan('2026-11-02','primary',Calendar());now=[p['window_start']-pd.Timedelta(seconds=3)]
    def sleep(seconds):assert seconds<=1;now[0]+=pd.Timedelta(seconds=seconds)
    d.arm(p,lambda:now[0],sleep)
    assert now[0]==p['window_start']
    with pytest.raises(ValueError,match='30 minutes'):d.arm(p,lambda:p['window_start']-pd.Timedelta(minutes=31),sleep)


def test_real_library_signatures_constructor_and_transport_offline(monkeypatch):
    import inspect,yfinance as yf,curl_cffi
    from curl_cffi.requests import Session
    from yfinance.scrapers.history import PriceHistory
    assert yf.__version__=='1.7.0' and curl_cffi.__version__=='0.16.3'
    assert 'session' in inspect.signature(yf.Ticker).parameters
    for parameter in ['end','keepna','prepost','auto_adjust']:
        assert parameter in inspect.signature(PriceHistory.history).parameters
    for parameter in ['allow_redirects','max_redirects','timeout']:
        assert parameter in inspect.signature(Session.request).parameters
    now=pd.Timestamp('2026-11-02T22:00:00Z');seen=[]
    def fake_once(self,method,url,**kwargs):
        seen.append(kwargs);return type('Response',(),{'status_code':200})()
    monkeypatch.setattr(Session,'_request_once',fake_once)
    transport=d.bounded_session_class(Session,lambda:now,now+pd.Timedelta(minutes=15))(impersonate='chrome',retry=0)
    try:
        ticker=yf.Ticker('SPY',session=transport)
        assert ticker.session is transport
        transport.get('https://example.invalid')
        assert transport.retry.count==0 and transport.calls==1
        assert seen[0]['allow_redirects'] is False and seen[0]['max_redirects']==0
        transport.metadata_only=True
        with pytest.raises(RuntimeError,match='Metadata-only'):transport.get('https://example.invalid')
        assert transport.calls==1
    finally:transport.close()


@pytest.mark.parametrize('session', ['2026-11-02','2026-11-06','2026-11-25','2026-11-27','2026-10-30'])
@pytest.mark.parametrize('interval',['1d','1wk','1mo'])
def test_real_yfinance_reference_end_filters_next_date_offline(monkeypatch,session,interval):
    import yfinance as yf
    from yfinance.data import YfData
    from yfinance.base import TickerBase
    from curl_cffi.requests import Session
    next_day=date.fromisoformat(session)+pd.Timedelta(days=1)
    target=pd.Timestamp(session+' 09:30',tz=d.NY)
    cap=pd.Timestamp(str(next_day)+' 00:00',tz=d.NY)
    # Fabricated chart payload exercises actual pinned parser and explicit-end cleanup.
    stamps=[int(target.timestamp()),int((cap+pd.Timedelta(hours=8)).timestamp())]
    chart={'chart':{'error':None,'result':[{'timestamp':stamps,'meta':{
        'currency':'USD','instrumentType':'ETF','exchangeTimezoneName':d.NY,'validRanges':['1mo'],'priceHint':2},
        'indicators':{'quote':[{'open':[100.,100.],'high':[110.,110.],'low':[90.,90.],'close':[105.,106.],'volume':[1000,2000]}],
                      'adjclose':[{'adjclose':[104.,105.]}]}}]}}
    requests=[]
    class Response:
        text='fixture'
        def json(self):return deepcopy(chart)
    def fake_get(self,*a,**k):requests.append(k);return Response()
    monkeypatch.setattr(YfData,'get',fake_get);monkeypatch.setattr(YfData,'cache_get',fake_get)
    monkeypatch.setattr(TickerBase,'_get_ticker_tz',lambda *a,**k:d.NY)
    monkeypatch.setattr(Session,'request',lambda *a,**k:pytest.fail('live HTTP forbidden'))
    with Session(impersonate='chrome',retry=0) as transport:
        ticker=yf.Ticker('SPY',session=transport)
        result=ticker.history(start=target-pd.Timedelta(days=90),end=cap,interval=interval,auto_adjust=False,actions=False,prepost=False,repair=False,keepna=True)
        assert result.index[-1].date()==date.fromisoformat(session)
        assert result.iloc[-1]['Close']==105.
        assert requests[0]['params']['period2']==int(cap.timestamp())
        # Supported getter returns base metadata without another chart request.
        count=len(requests);view=ticker.get_history_metadata(repair=False)
        assert view['exchangeTimezoneName']==d.NY and len(requests)==count


def test_absent_adjusted_close_allows_core_comparison(tmp_path):
    reports=[]
    for stage in ['primary','later','reference']:
        p=d.plan('2026-11-02',stage,Calendar());f=frame();f['Adj Close']=None
        r=d.collect([{'symbol':'SPY'}],p,lambda *a:(f,meta()),lambda:p['window_start'],tmp_path/stage,lambda:3,{},'a'*64)
        reports.append(r)
    row=d.compare(reports)['comparisons'][0]
    assert row['status']=='CORE_STABLE_ADJ_CLOSE_UNAVAILABLE'
    assert row['core_comparison_eligible'] and not row['adj_close_comparison_eligible']


def test_holiday_target_rejected_and_half_day_reference_bound():
    class Holiday(Calendar):
        def is_session(self,t):return t.date()!=date(2026,11,26) and super().is_session(t)
        def session_close(self,t):return pd.Timestamp(str(t.date())+' 13:00',tz=d.NY)
    with pytest.raises(ValueError,match='not an XNYS session'):d.plan('2026-11-26','primary',Holiday())
    p=d.plan('2026-11-27','reference',Holiday())
    assert p['request_end_cap']==pd.Timestamp('2026-11-28 00:00',tz=d.NY)


def test_legitimate_container_metadata_does_not_crash_capture(tmp_path):
    p=d.plan('2026-11-02','primary',Calendar());m=meta();m['lastTrade']=[None,True,{'foo':'bar'}]
    m['currentTradingPeriod']={'regular':[{'start':1,'end':2}], 'api_key':'SECRET'}
    r=d.collect([{'symbol':'SPY'}],p,lambda *a:(frame(),m),lambda:p['window_start'],tmp_path/'meta',lambda:3,{},'a'*64)
    assert r['collection_complete'] and r['decision']=='NO_GO_OR_UNVERIFIED'
    assert 'SECRET' not in (tmp_path/'meta'/'observations.json').read_text()


@pytest.mark.parametrize('field,value',[('Open',float('inf')),('Volume',-1.),('Low',111.),('Close',89.)])
def test_required_ohlcv_failures(field,value):
    f=frame();f[field]=value
    assert d.assess(f,meta(),'1d',d.plan('2026-11-02','primary',Calendar()))['status']=='INVALID_OHLCV'


def test_public_metadata_getter_cannot_add_requests():
    now=pd.Timestamp('2026-11-02T22:00:00Z')
    class Base:
        def request(self,*a,**k):pytest.fail('metadata caused HTTP')
    transport=d.bounded_session_class(Base,lambda:now,now+pd.Timedelta(minutes=15))()
    class Ticker:
        def get_history_metadata(self,repair=False):
            transport.request('GET','https://example.invalid')
    assert d.observed_metadata(Ticker(),transport)=={}
    assert transport.calls==0 and not transport.metadata_only
