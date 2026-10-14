from __future__ import annotations

import copy
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import signal
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from diagnostics import stock_eod_acquisition_capture as diag
from etl import sources, eod_family
from core import storage, stock_eod_family as publication

RUN = "prod-20261013T220000Z-abcdef12"


def frame():
    data = pd.DataFrame({"Open": [10., 10., 10.], "High": [12., 12., 12.],
                         "Low": [9., 9., 9.], "Close": [11., 11., np.nan],
                         "Adj Close": [11., 11., np.nan], "Volume": [100., 100., 100.]},
                        index=pd.DatetimeIndex(["2026-10-12", "2026-10-12", "2026-10-13"], tz="America/New_York", name="Date"))
    data.attrs = {"secret": "SECRET-cookie-token", "eod_provenance": {
        "source_interval": "1d", "session_semantics": "regular", "adjustment_mode": "raw_unadjusted",
        "provider_metadata": {"lastTrade": {"Price": 11., "Time": pd.Timestamp("2026-10-13")},
                              "cookie": "SECRET-cookie-token", "regularMarketPrice": 11.}}}
    return data


def config(tmp_path, **kwargs):
    return diag.Config(application=True, output_dir=str(tmp_path), selected_symbols=("COF",), **kwargs)


def install_ticker(monkeypatch, calls, returned, missing_close=False):
    class Ticker:
        def __init__(self, symbol): self.symbol = symbol
        def history(self, **kwargs):
            # Fixed timestamps make actual parameters and frame bytes comparable.
            calls.append(("history", self.symbol, kwargs))
            result = frame()
            if kwargs['interval'] != '1d':
                result.loc[:, 'Close'] = 11.
                result.loc[:, 'Adj Close'] = 11.
            if missing_close: result = result.drop(columns=['Close'])
            returned.append((result, result.copy(deep=True), copy.deepcopy(result.attrs)))
            return result
        def get_history_metadata(self):
            calls.append(("metadata", self.symbol))
            return {"priceHint": 2, "lastTrade": {"Price": 11., "Time": pd.Timestamp('2026-10-13')},
                    "cookie": "SECRET-cookie-token", "exchangeTimezoneName": "America/New_York"}
    monkeypatch.setattr(sources.yf, 'Ticker', Ticker)


def collect_family(cfg, calls, returned, monkeypatch):
    install_ticker(monkeypatch, calls, returned)
    with diag.collecting(cfg) as collector:
        family = eod_family.acquire_stock_eod_family(RUN, 'COF', loader=sources.safe_load_eod)
        finalized = eod_family.reconcile_stock_eod_family(family)
        manifest = eod_family.family_manifest(finalized)
    return family, finalized, manifest, collector


def test_application_on_preserves_history_frames_requests_and_policy(tmp_path, monkeypatch):
    class Clock:
        @staticmethod
        def now(tz=None): return pd.Timestamp('2026-10-13T22:00:00Z')
    class PandasProxy:
        Timestamp=Clock
        def __getattr__(self,name): return getattr(pd,name)
    monkeypatch.setattr(sources,'pd',PandasProxy())
    results = []
    for enabled in [False, True]:
        calls, returned = [], []
        cfg = diag.Config(application=enabled, output_dir=str(tmp_path / str(enabled)), selected_symbols=('COF',))
        family, finalized, manifest, collector = collect_family(cfg, calls, returned, monkeypatch)
        results.append((family, finalized, manifest, calls, collector))
        # Existing adapter intentionally mutates the returned frame's index.
        # Compare those post-adapter side effects with diagnostics OFF.
        if enabled:
            for (observed,_,_), (baseline,_,_) in zip(returned,baseline_returned):
                pd.testing.assert_frame_equal(observed,baseline)
                assert observed.attrs==baseline.attrs
        else: baseline_returned=returned
    assert results[0][3] == results[1][3]
    for a, b in zip(results[0][3], results[1][3]):
        if a[0] == 'history':
            assert {k:v for k,v in a[2].items() if k not in {'start','end'}} == {k:v for k,v in b[2].items() if k not in {'start','end'}}
    assert results[0][0].requests == results[1][0].requests
    for tf in ['daily','weekly','monthly']:
        a,b = getattr(results[0][1],tf),getattr(results[1][1],tf)
        assert a.data_quality_status == b.data_quality_status
        pd.testing.assert_frame_equal(a.finalized_frame,b.finalized_frame)
    # Timings and payload provenance fetch times naturally differ between executions.
    assert results[0][1].reconciliation == results[1][1].reconciliation
    records = [json.loads(x) for x in results[1][4].records['selected']]
    assert len(records)==3
    assert [r['identity']['native_interval'] for r in records]==['1d','1wk','1mo']
    assert [r['identity']['acquisition_sequence'] for r in records]==[1,2,3]
    assert records[0]['boundaries']['B']['rows']==3
    assert records[0]['boundaries']['C']['rows']==2
    assert records[0]['stage_facts']['C']['duplicates_removed']['value']==1
    assert 'SECRET-cookie-token' not in json.dumps(records)
    assert not list((tmp_path/'False').glob('*')) if (tmp_path/'False').exists() else True


@pytest.mark.parametrize('value,kind',[(None,'None'),(pd.NA,'pd.NA'),(np.nan,'NaN'),(np.inf,'+infinity'),(-np.inf,'-infinity'),(1.25,'finite')])
def test_exact_numeric_types(value,kind):
    assert diag.atom(value)['kind']==kind
    if kind=='finite': assert float.fromhex(diag.atom(value)['hex'])==value


def test_missing_column_and_schema_changes(tmp_path,monkeypatch):
    calls,returned=[],[]
    install_ticker(monkeypatch,calls,returned,missing_close=True)
    with diag.collecting(config(tmp_path)) as collector:
        eod_family.acquire_stock_eod_family(RUN,'COF',loader=sources.safe_load_eod)
    record=json.loads(collector.records['selected'][0])
    assert 'close' in record['boundaries']['B']['absent_quote_columns']
    assert record['stage_facts']['C']['inserted_columns']==['close']
    assert record['boundaries']['B']['terminal_rows'][-1]['label']['value'].endswith('-04:00')
    assert record['boundaries']['C']['terminal_rows'][-1]['label']['value']=='2026-10-13T00:00:00'


def test_raw_request_is_disabled_without_hooks_and_preserves_cache_retry_identity(tmp_path):
    # Layer 2 is not installed: shared YfData methods/cache/response remain original.
    from yfinance.data import YfData
    originals=(YfData.get,YfData.cache_get)
    events=[]
    response=object()
    def original_get(key):
        events.extend([('attempt',key,1),('retry',key,2)])
        return response
    cached=lru_cache()(original_get)
    with diag.collecting(diag.Config(raw_response=True,output_dir=str(tmp_path))) as collector:
        with diag.acquisition(RUN,'COF','1wk'):
            assert cached('key') is response
            assert cached('key') is response
    assert collector.raw_status=='get_cache_get_seam_does_not_expose_internal_retry_lineage'
    assert events==[('attempt','key',1),('retry','key',2)]
    assert (YfData.get,YfData.cache_get)==originals
    assert diag._FETCH.get() is None and diag._ACTIVE.get() is None


def test_unsupported_raw_version_and_source_dont_disable_application(tmp_path,monkeypatch):
    import yfinance
    with monkeypatch.context() as patch:
        patch.setattr(yfinance,'__version__','unsupported')
        with diag.collecting(config(tmp_path,raw_response=True)) as collector:
            assert collector.raw_status=='unsupported_yfinance_version'
            assert collector.config.application
    patch_source=lambda f:'different source'
    monkeypatch.setattr(diag.inspect,'getsource',patch_source)
    with diag.collecting(config(tmp_path,raw_response=True)) as collector:
        assert collector.raw_status=='unsupported_yfinance_source'


def test_capture_and_serialization_failures_do_not_change_acquisition(tmp_path,monkeypatch):
    calls,returned=[],[]
    monkeypatch.setattr(diag,'snapshot',lambda *a:(_ for _ in ()).throw(ValueError('capture failure')))
    family,finalized,_,collector=collect_family(config(tmp_path),calls,returned,monkeypatch)
    assert family.requests.succeeded==3
    assert finalized.daily.data_quality_status=='repaired'
    assert collector.gaps['capture_error_or_noop']==6
    with monkeypatch.context() as patch:
        patch.setattr(diag.Collector,'complete',lambda *a,**k:(_ for _ in ()).throw(ValueError('encoding failure')))
        collect_family(config(tmp_path),[],[],monkeypatch)


def test_snapshot_itself_leaves_frame_and_attrs_untouched():
    observed=frame();before=observed.copy(deep=True);attrs=copy.deepcopy(observed.attrs)
    diag.snapshot(observed)
    pd.testing.assert_frame_equal(observed,before)
    assert observed.attrs==attrs


def test_genuine_timeout_and_base_exceptions_propagate_and_context_restores(tmp_path):
    for exc in [sources.TimeoutException('alarm'),TimeoutError('alarm'),KeyboardInterrupt(),SystemExit(2)]:
        with pytest.raises(type(exc)):
            with diag.collecting(config(tmp_path)):
                with diag.acquisition(RUN,'COF','1d'):
                    diag.attempt(lambda:(_ for _ in ()).throw(exc))
        assert diag._FETCH.get() is None and diag._ACTIVE.get() is None


def test_sigalrm_during_snapshot_keeps_original_loader_timeout_contract(tmp_path,monkeypatch):
    calls,returned=[],[]
    install_ticker(monkeypatch,calls,returned)
    def interrupted(*args):
        signal.raise_signal(signal.SIGALRM)
    monkeypatch.setattr(diag,'snapshot',interrupted)
    with diag.collecting(config(tmp_path)):
        family=eod_family.acquire_stock_eod_family(RUN,'COF',loader=sources.safe_load_eod)
    assert family.requests.failed==3
    assert all(x.acquisition_status=='failed' for x in family.observations())


def test_limits_redaction_and_local_storage_only(tmp_path,monkeypatch):
    def forbidden(*a,**k):pytest.fail('diagnostic used production write API')
    for name in ['save_parquet','compare_and_swap_bytes','create_bytes_if_absent','delete_s3_prefix']:
        monkeypatch.setattr(storage,name,forbidden)
    calls,returned=[],[]
    cfg=config(tmp_path,slots={k:0 for k in diag.Config().slots})
    family,_,_,collector=collect_family(cfg,calls,returned,monkeypatch)
    assert family.requests.succeeded==3
    assert collector.gaps['selected_quota']==1
    assert not any(collector.records.values())
    cfg=config(tmp_path,per_fetch_bytes=256)
    family,_,_,collector=collect_family(cfg,[],[],monkeypatch)
    assert family.requests.succeeded==3 and collector.gaps['oversized_fetch']==3
    wide=pd.DataFrame(np.ones((2,33)))
    assert diag.snapshot(wide)['reason']=='too_many_columns'
    evidence=next(tmp_path.glob('acquisition-*'))
    assert evidence.stat().st_mode & 0o777==0o700
    assert (evidence/'summary.json').stat().st_mode & 0o777==0o600


def test_reserved_categories_preserve_controls_and_complete_families(tmp_path):
    def acquired(symbol,bad):
        loader=lambda *a,**k: frame().rename(columns=str.lower).assign(close=np.nan if bad else 11.)
        # Synthetic loader explicitly marks the two observations, without provider calls.
        def observed(*a,**k):
            f=loader();diag.capture('B',f);diag.capture('C',f);return f
        return eod_family.acquire_stock_eod_family(RUN,symbol,loader=observed)
    cfg=diag.Config(application=True,output_dir=str(tmp_path),boundary_seconds=0,background_modulus=1,
                    slots={'selected':0,'boundary':0,'malformed':3,'controls':6,'background':3})
    with diag.collecting(cfg) as collector:
        acquired('BEFORE',False);acquired('BAD',True);acquired('BAD2',True);acquired('AFTER',False)
    assert collector.counts['malformed']==3
    assert collector.gaps['malformed_quota']==1
    assert collector.counts['controls']==6
    assert {json.loads(r)['identity']['symbol'] for r in collector.records['controls']}=={'BEFORE','AFTER'}


def test_deterministic_production_outputs_identical_with_capture(tmp_path,monkeypatch):
    # Reuse the existing complete production fixture (reference requests, repaired
    # daily, excluded weekly, indicators, snapshots, immutable publication).
    path=Path(__file__).with_name('test_stock_eod_family_production_integration.py')
    spec=importlib.util.spec_from_file_location('production_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.acquire_stock_eod_family
    def observing(run_id,symbol,*,loader,**kwargs):
        def wrapped_loader(*a,**k):
            result=loader(*a,**k);diag.capture('B',result);diag.capture('C',result);return result
        return original(run_id,symbol,loader=wrapped_loader,**kwargs)
    module.acquire_stock_eod_family=observing
    outputs=[]
    for enabled in [False,True]:
        root=tmp_path/str(enabled);root.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(publication,'_utc_iso',lambda:'2026-10-13T22:00:00Z')
            with diag.collecting(diag.Config(application=enabled,raw_response=enabled,selected_symbols=('AAPL','MSFT'))) as collector:
                module.test_authoritative_family_handoff_is_provider_closed_and_publishes_finalized_frames(patch,root)
            # Published generation/pointer bytes exist on disk; rolling snapshots
            # are kept in the fixture's actual storage dictionary.
            files={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file() and not p.name.endswith('.lock')}
            load=module.runner.storage.load_parquet
            snapshots=[load(root/name) for name in ['snapshot_stocks_daily.parquet','snapshot_stocks_weekly.parquet','snapshot_stocks_monthly.parquet']]
            outputs.append((files,snapshots,collector))
    assert outputs[0][0]==outputs[1][0]
    for a,b in zip(outputs[0][1],outputs[1][1]):
        pd.testing.assert_frame_equal(a,b)
        assert a.to_parquet()==b.to_parquet()
    assert outputs[1][2].counts['selected']==6


def test_timing_memory_bound_and_budget_accounting(tmp_path):
    f=frame();cfg=config(tmp_path)
    c=diag.Collector(cfg);times=[]
    tracemalloc.start()
    for n in range(300):
        fetch=c.begin(RUN,'COF','1d');token=diag._FETCH.set(fetch)
        start=time.perf_counter_ns();c.capture('B',f,{});c.capture('C',f,{});times.append(time.perf_counter_ns()-start)
        diag._FETCH.reset(token);c.complete(fetch)
        # Exercise backlog bound rather than retaining unlimited references.
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert c.pending_bytes<=cfg.memory_bytes//4
    assert peak<cfg.memory_bytes
    assert max(times)<100_000_000  # conservative offline CI guard, not runtime promise
    print(f'MEASUREMENT B+C median_ms={np.median(times)/1e6:.3f} max_ms={max(times)/1e6:.3f} peak_bytes={peak}')


def test_default_environment_is_off_and_workflow_is_nonblocking(monkeypatch,tmp_path):
    import yaml
    for key in ['STOCK_EOD_CAPTURE_APPLICATION','STOCK_EOD_CAPTURE_RAW','STOCK_EOD_CAPTURE_DIR']:
        monkeypatch.delenv(key,raising=False)
    @diag.configured_family
    def run():
        assert diag._ACTIVE.get() is None
        return 'unchanged'
    assert run()=='unchanged'
    wf=Path(__file__).resolve().parents[1]/'.github/workflows/stocks_daily_eod.yml'
    config=yaml.safe_load(wf.read_text());on=config.get('on',config.get(True))
    assert all(not x['default'] for x in on['workflow_dispatch']['inputs'].values())
    step=[s for s in config['jobs']['daily-eod-stocks']['steps'] if s.get('uses')=='actions/upload-artifact@v4'][0]
    assert step['continue-on-error'] is True
    assert step['with']['retention-days']==7


def test_capture_contexts_isolated_across_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def observe(symbol):
        with diag.collecting(diag.Config(application=True,selected_symbols=(symbol,))) as collector:
            with diag.acquisition(RUN,symbol,'1d'):
                diag.capture('B',frame())
            record=collector.pending[0]
            assert record['identity']['symbol']==symbol
            return record['identity']
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(observe,['A','B']))
    assert [r['symbol'] for r in results]==['A','B']
    assert [r['acquisition_sequence'] for r in results]==[1,1]
    assert diag._ACTIVE.get() is None


def test_diagnostic_output_failure_preserves_original_production_exception(monkeypatch,tmp_path):
    monkeypatch.setattr(diag.Collector,'finish',lambda *a:(_ for _ in ()).throw(OSError('disk full')))
    with diag.collecting(config(tmp_path)):
        pass
    with pytest.raises(RuntimeError,match='production failure'):
        with diag.collecting(config(tmp_path)):
            raise RuntimeError('production failure')


def test_facts_dont_leak_secrets_and_snapshot_is_detached(tmp_path):
    f=frame()
    f.attrs['eod_provenance']['yfinance_version']='SECRET-cookie-token'
    f.attrs['eod_provenance']['provider_metadata'].update({
        'range':'SECRET-cookie-token','exchangeTimezoneName':'SECRET-cookie-token',
        'dataGranularity':'SECRET-cookie-token','metadata_error':'SECRET-cookie-token'})
    with diag.collecting(config(tmp_path)) as c:
        with diag.acquisition(RUN,'COF','1d'):
            diag.capture('B',f,cookie='SECRET-cookie-token',headers={'Authorization':'SECRET-cookie-token'})
            f.iloc[-1,f.columns.get_loc('Close')]=123.
        r=c.pending[0]
        assert r['boundaries']['B']['terminal_rows'][-1]['values']['3']['kind']=='NaN'
        assert 'SECRET-cookie-token' not in json.dumps(r)
        assert r['stage_facts']['B']=={}
