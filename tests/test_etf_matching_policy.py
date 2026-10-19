"""Approved policy and complete committed-universe regression against main e2d4ee5.

The frozen matcher/configuration fixtures replay the original behavior independently
of the new tokenizer, candidate filter, and added symbol overrides.
"""
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from etl import etf_matching as matching
from etf import mapping_engine, trend_engine, guardrails, universe
from jobs import run_etf_trends as acquisition

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name("fixtures")
ASSOCIATIONS = ["etf_symbol_primary", "etf_symbol_secondary"]
RESIDENTIAL = set("CVCO DHI GRBK IBP KBH LEN MHO MTH PHM SKY TOL".split())
ENGINEERING = set(
    "ACA ACM AGX AMRC APG DY ECG EME EXPO FER FIX FLR GVA IESC KBR LGN "
    "MTZ MYRG PRIM ROAD STN STRL TPC TTEK WLDN".split()
)
MACHINERY = set("AGCO ALG BLBD CAT CNH DE FSS LNN OSK PCAR TEX".split())
ELECTRONICS = {"LPL", "SONO", "SONY"}
SCHEMA = [
    "symbol", "name", "sector", "industry", "industry_etf", "industry_etf_name",
    "industry_score", "sector_etf", "sector_etf_name", "sector_score",
    "etf_symbol_primary", "etf_symbol_secondary", "etf_match_source", "etf_override_notes",
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ETF policy tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize(("text", "expected"), [
    ("S&P U.S. s&p u.s.", set()),
    ("  SPDR\tS&P\nSoftware & SERVICES ETF  ", {"software"}),
    ("alpha-S&P-beta", {"alpha", "beta"}),
    ("alpha---U.S.---beta", {"alpha", "beta"}),
    ("alpha & beta", {"alpha", "beta"}),
    ("firsthand servicescape spdrx", {"firsthand", "servicescape", "spdrx"}),
    ("XS&PX", {"xs", "px"}),
    ("AU.S.B", {"au", "s", "b"}),
    ("U.S.A S&P500", {"u", "s", "a", "p500"}),
    ("S & P U. S.", {"s", "p", "u"}),
    ("Healthcare Technologies Industrials Utilities", {"health", "technology", "industrial", "utility"}),
    ("Financial Data & Stock Exchanges", {"financial", "data", "stock"}),
    ("Exchanges", set()),
    ("Oil & Gas", {"oil", "gas"}),
    ("THE first AND SPDR ETF Fund Trust", set()),
    (None, set()), (42, set()), ("", set()),
])
def test_tokenizer(text, expected):
    assert matching._tokenize(text) == expected


@pytest.mark.parametrize("value", [None, np.nan, pd.NA, "", "  ", "true", " TRUE ", True])
def test_auto_match_default_and_true(value):
    etfs = pd.DataFrame({"symbol": ["TEST"], "name": ["Test"], "auto_match": [value]})
    pd.testing.assert_frame_equal(mapping_engine._automatic_matching_candidates(etfs), etfs)


@pytest.mark.parametrize("value", ["false", " FALSE ", False])
def test_auto_match_false_and_order(value):
    etfs = pd.DataFrame({"symbol": ["Z", "X", "A"], "auto_match": ["", value, "true"]}, index=[8, 2, 5])
    original = etfs.copy(deep=True)
    selected = mapping_engine._automatic_matching_candidates(etfs)
    assert selected.symbol.tolist() == ["Z", "A"]
    assert selected.index.tolist() == [8, 5]
    pd.testing.assert_frame_equal(etfs, original)


def test_missing_auto_match_column():
    etfs = pd.DataFrame({"symbol": ["Z", "A"]})
    pd.testing.assert_frame_equal(mapping_engine._automatic_matching_candidates(etfs), etfs)


@pytest.mark.parametrize("value", ["yes", "no", "1", "0", 1, 0, "disabled", "tru"])
def test_invalid_auto_match_identifies_field_and_etf(value):
    etfs = pd.DataFrame({"symbol": ["ITB"], "auto_match": [value]})
    with pytest.raises(ValueError, match="auto_match.*ITB"):
        mapping_engine._automatic_matching_candidates(etfs)


@pytest.mark.parametrize("value", ["NA", "null", "nan", "N/A", "yes", "1"])
def test_csv_loader_preserves_invalid_nonblank_flags(tmp_path, monkeypatch, value):
    (tmp_path / "shortlist_sector_etfs.csv").write_text(
        f"symbol,name,auto_match\nITB,Home Construction,{value}\n"
    )
    monkeypatch.setattr(universe, "CFG", tmp_path)
    etfs = universe.load_etf_universe()
    assert etfs.auto_match.iloc[0] == value
    with pytest.raises(ValueError, match="auto_match.*ITB"):
        mapping_engine._automatic_matching_candidates(etfs)


@pytest.mark.parametrize("header,value,expected", [
    ("symbol,name", "ITB,Home Construction", ["ITB"]),
    ("symbol,name,auto_match", "ITB,Home Construction,", ["ITB"]),
    ("symbol,name,auto_match", "ITB,Home Construction, TRUE ", ["ITB"]),
    ("symbol,name,auto_match", "ITB,Home Construction, FALSE ", []),
])
def test_csv_loader_flag_defaults(tmp_path, monkeypatch, header, value, expected):
    (tmp_path / "shortlist_sector_etfs.csv").write_text(f"{header}\n{value}\n")
    monkeypatch.setattr(universe, "CFG", tmp_path)
    assert mapping_engine._automatic_matching_candidates(universe.load_etf_universe()).symbol.tolist() == expected


@pytest.fixture
def universe_replay(tmp_path):
    namespace = {"__name__": __name__, "__file__": str(ROOT / "etl/etf_matching.py")}
    source = FIXTURES / "etf_matching_original.py.txt"
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    (tmp_path / "etf_match_overrides.csv").write_bytes((FIXTURES / "etf_overrides_original.csv").read_bytes())
    namespace["CFG"] = tmp_path
    stocks = pd.read_csv(ROOT / "ref/options_eligible.csv").dropna(subset=["sector", "industry"])
    original_etfs = pd.read_csv(FIXTURES / "etf_shortlist_original.csv")
    before = namespace["build_symbol_to_etf_map"](stocks, original_etfs).fillna("")
    after = mapping_engine.build_options_eligible_etf_map().fillna("")
    return namespace, stocks, before, after


def test_complete_universe_approved_diff(universe_replay):
    original, stocks, before, after = universe_replay
    assert len(pd.read_csv(ROOT / "ref/options_eligible.csv")) == 2325
    assert len(stocks) == len(before) == len(after) == 2322
    assert list(after) == list(before) == SCHEMA
    assert not after.symbol.duplicated().any()
    pd.testing.assert_frame_equal(before[["symbol", "name", "sector", "industry"]], after[["symbol", "name", "sector", "industry"]])
    primary = before.etf_symbol_primary.ne(after.etf_symbol_primary)
    secondary = before.etf_symbol_secondary.ne(after.etf_symbol_secondary)
    assert (int(primary.sum()), int(secondary.sum()), int((primary & secondary).sum())) == (28, 25, 25)
    assert int((~(primary | secondary)).sum()) == 2294
    assert set(after.loc[primary | secondary, "symbol"]) == ENGINEERING | ELECTRONICS
    assert not after.etf_symbol_primary.eq("").any()
    rows = after.set_index("symbol")
    for symbols, expected in [
        (RESIDENTIAL, ["ITB", "RSPD"]), (ENGINEERING | {"J", "PWR"}, ["XLI", ""]),
        (MACHINERY, ["XLI", ""]), (ELECTRONICS, ["RSPD", "XLK"]), ({"ACN"}, ["XSW", "XLK"]),
    ]:
        assert rows.loc[sorted(symbols), ASSOCIATIONS].values.tolist() == [expected] * len(symbols)
    assert set(stocks.loc[stocks.industry.eq("Residential Construction"), "symbol"]) == RESIDENTIAL
    assert set(stocks.loc[stocks.industry.eq("Engineering & Construction"), "symbol"]) == ENGINEERING | {"J", "PWR"}
    assert set(stocks.loc[stocks.industry.eq("Farm & Heavy Construction Machinery"), "symbol"]) == MACHINERY
    assert rows.loc[sorted(RESIDENTIAL | {"J", "PWR", "ACN"}), "etf_match_source"].eq("override").all()
    assert rows.loc[sorted(ENGINEERING | MACHINERY | ELECTRONICS), "etf_match_source"].eq("fuzzy").all()
    software = before.etf_symbol_primary.eq("IGV")
    assert int(software.sum()) == 170
    pd.testing.assert_frame_equal(before.loc[software, ASSOCIATIONS], after.loc[software, ASSOCIATIONS])
    assert after.loc[software, "etf_symbol_secondary"].eq("XLK").all()
    existing = before.etf_match_source.eq("override")
    assert int(existing.sum()) == 21
    pd.testing.assert_frame_equal(before.loc[existing, ASSOCIATIONS], after.loc[existing, ASSOCIATIONS])
    prior_overrides = original["_load_etf_match_overrides"]()
    overrides = matching._load_etf_match_overrides()
    assert len(prior_overrides) == 22 and len(overrides) == 33
    assert {key: overrides[key] for key in prior_overrides} == prior_overrides
    assert set(overrides) - set(prior_overrides) == RESIDENTIAL
    for symbol in RESIDENTIAL:
        assert overrides[symbol] == {"etf_symbol_primary": "ITB", "etf_symbol_secondary": "RSPD", "notes": "manual override"}
    assert matching._STOPWORDS == original["_STOPWORDS"]
    assert "services" in matching._STOPWORDS
    etfs = pd.read_csv(ROOT / "config/shortlist_sector_etfs.csv")
    pd.testing.assert_frame_equal(etfs[["symbol", "name"]], pd.read_csv(FIXTURES / "etf_shortlist_original.csv"))
    selected = mapping_engine._automatic_matching_candidates(etfs)
    assert len(etfs) == 38 and len(selected) == 36
    assert selected.symbol.tolist() == [symbol for symbol in etfs.symbol if symbol not in {"ITB", "XSW"}]


def test_scores_thresholds_ties_and_override_acceptance(monkeypatch):
    assert matching._jaccard({"software", "application"}, {"software", "tech"}) == 1 / 3
    etfs = pd.DataFrame({"symbol": ["FIRST", "SECOND"], "name": ["software", "software"]})
    stocks = pd.DataFrame({"symbol": ["TEST"], "industry": ["software"], "sector": ["software"]})
    monkeypatch.setattr(matching, "_load_etf_match_overrides", lambda: {})
    assert matching.build_symbol_to_etf_map(stocks, etfs).iloc[0].etf_symbol_primary == "FIRST"
    assert matching.build_symbol_to_etf_map(stocks, etfs.iloc[::-1]).iloc[0].etf_symbol_primary == "SECOND"
    for score, expected in [(.199999, "SECOND"), (.2, "FIRST")]:
        scores = iter([score, 0., 0., .5])
        monkeypatch.setattr(matching, "_jaccard", lambda *_: next(scores))
        assert matching.build_symbol_to_etf_map(stocks, etfs).iloc[0].etf_symbol_primary == expected
    monkeypatch.setattr(matching, "_jaccard", lambda *_: 0.)
    monkeypatch.setattr(matching, "_load_etf_match_overrides", lambda: {
        "TEST": {"etf_symbol_primary": "XSW", "etf_symbol_secondary": "ITB", "notes": "manual override"},
    })
    row = matching.build_symbol_to_etf_map(stocks, etfs).iloc[0]
    assert row[ASSOCIATIONS].tolist() == ["XSW", "ITB"]
    assert row.etf_match_source == "override"


@pytest.mark.parametrize("timeframe", ["daily", "weekly", "monthly", "quarterly"])
def test_acquisition_scores_cache_and_joins_ignore_auto_match(monkeypatch, timeframe):
    expected = set(pd.read_csv(ROOT / "config/shortlist_sector_etfs.csv").symbol)
    assert set(acquisition.load_etf_symbols()) == expected
    assert {"XSW", "ITB"}.issubset(expected)
    saved, requested = {}, []
    def provider(symbol, **kwargs):
        requested.append(symbol)
        return pd.DataFrame({
            "open": [100.], "high": [102.], "low": [99.], "close": [101.],
            "adj_close": [101.], "volume": [1000.], "wyckoff_stage": [2.],
            "macdv_guard": [0.], "sig_vol_current_bar": [1.],
        }, index=pd.DatetimeIndex(["2026-10-16"], name="date"))
    monkeypatch.setattr(acquisition, "load_eod", provider)
    monkeypatch.setattr(acquisition, "initialize_indicator_engine", lambda *_: None)
    monkeypatch.setattr(acquisition, "apply_core", lambda frame, **_: frame)
    monkeypatch.setattr(acquisition.storage, "exists", lambda _: False)
    monkeypatch.setattr(acquisition.storage, "save_parquet", lambda frame, path: saved.update({str(path): frame.copy()}))
    acquisition.ingest_etf_timeframe(timeframe)
    assert set(requested) == expected
    snapshot = saved[str(acquisition.DATA / f"snapshot_etf_{timeframe}.parquet")]
    assert set(snapshot.symbol) == expected
    monkeypatch.setattr(trend_engine.storage, "exists", lambda _: True)
    monkeypatch.setattr(trend_engine.storage, "load_parquet", lambda _: snapshot)
    scores = trend_engine.compute_etf_trend_scores(timeframe)
    assert set(scores.index) == expected
    monkeypatch.setattr(trend_engine.storage, "load_parquet", lambda _: scores)
    pd.testing.assert_frame_equal(trend_engine.load_etf_trend_scores(timeframe), scores)
    mapping = mapping_engine.build_options_eligible_etf_map()
    original_read = pd.read_csv
    monkeypatch.setattr(guardrails.pd, "read_csv", lambda path, *args, **kwargs:
                        mapping.copy() if str(path).endswith("symbol_to_etf_options_eligible.csv")
                        else original_read(path, *args, **kwargs))
    monkeypatch.setattr(guardrails, "load_etf_trend_scores", lambda _: scores)
    joined = guardrails.attach_etf_trends_for_options_combo(
        mapping[["symbol"]], {"universe": "options_eligible"}, lower_tf=timeframe, middle_tf=timeframe,
    ).set_index("symbol")
    assert len(joined) == 2322 and not joined.index.duplicated().any()
    for symbol in RESIDENTIAL | {"ACN"}:
        assert joined.loc[symbol, [
            "etf_lower_primary_long_score", "etf_lower_secondary_long_score",
            "etf_middle_primary_long_score", "etf_middle_secondary_long_score",
        ]].tolist() == [5., 5., 5., 5.]
