import math
import sys
from collections.abc import Mapping

import pandas as pd
import pytest

from diagnostics.compare_native_derived_eod import (
    canonicalize,
    classify_observation,
    close_differences,
    corroborates_completion,
    derive_terminal,
    parse_symbols,
    payload_hash,
    price_tolerance,
    compare_symbol,
    json_safe,
    main,
)


def test_canonicalize_normalizes_timezone_and_preserves_raw_adjusted_close():
    frame = pd.DataFrame(
        {"Open": [1], "High": [3], "Low": [1], "Close": [2], "Adj Close": [1.5], "Volume": [4]},
        index=pd.DatetimeIndex(["2026-09-21 04:00:00+00:00"]),
    )
    result = canonicalize(frame)
    assert result.index[0] == pd.Timestamp("2026-09-21 00:00:00")
    assert result.iloc[0].to_dict() == {
        "open": 1.0, "high": 3.0, "low": 1.0, "close": 2.0,
        "adj_close": 1.5, "volume": 4.0,
    }


def test_derive_partial_week_and_month_through_latest_daily_session():
    index = pd.to_datetime(["2026-09-18", "2026-09-21", "2026-09-22"])
    daily = pd.DataFrame(
        {"open": [90, 100, 103], "high": [99, 105, 108], "low": [89, 98, 102],
         "close": [98, 104, 107], "adj_close": [98, 104, 107], "volume": [10, 20, 30]},
        index=index,
    )
    assert derive_terminal(daily, "1wk").to_dict() == {
        "open": 100, "high": 108, "low": 98, "close": 107, "adj_close": 107, "volume": 50,
    }
    assert derive_terminal(daily, "1mo").to_dict() == {
        "open": 90, "high": 108, "low": 89, "close": 107, "adj_close": 107, "volume": 60,
    }


def _interval(close, corroborated=True, tolerance=0.005):
    return {"status": "ok", "terminal": {"close": close}, "price_tolerance": tolerance,
            "completion_corroboration": corroborated}


def test_classification_requires_non_close_completion_evidence():
    intervals = {"1d": _interval(100), "1wk": _interval(100, False), "1mo": _interval(100)}
    assert classify_observation(intervals)["classification"] == "cannot establish comparable terminal session"


def test_classifies_exact_tolerant_disagreement_and_malformed():
    exact = {name: _interval(100) for name in ("1d", "1wk", "1mo")}
    assert classify_observation(exact)["classification"] == "D/W/M exact agreement"
    tolerant = {"1d": _interval(100), "1wk": _interval(100.004), "1mo": _interval(100.002)}
    assert classify_observation(tolerant)["classification"] == "agreement within deterministic tolerance"
    disagreement = {"1d": _interval(100), "1wk": _interval(100.02), "1mo": _interval(100)}
    assert classify_observation(disagreement)["classification"] == "disagreement"
    exact["1mo"] = {"status": "empty"}
    assert classify_observation(exact)["classification"] == "malformed/missing interval"


def test_completion_corroboration_excludes_close_but_requires_ohlv():
    native = {"open": 10, "high": 15, "low": 9, "close": 99, "volume": 100}
    derived = {"open": 10, "high": 15, "low": 9, "close": 12, "volume": 100}
    assert corroborates_completion(native, derived, 0.005)
    native["volume"] = 99
    assert not corroborates_completion(native, derived, 0.005)


def test_differences_and_provider_precision_tolerance():
    differences = close_differences({"1d": 100.0, "1wk": 100.01, "1mo": None})
    assert differences["1d_vs_1wk"]["absolute"] == pytest.approx(0.01)
    assert differences["1d_vs_1mo"] == {"absolute": None, "relative": None}
    assert price_tolerance({"priceHint": 2}) == 0.005


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"priceHint": 2}, 0.005),
        ({"priceHint": "3"}, 0.0005),
        ({}, 0.00005),
        ("{'priceHint': 2}", 0.00005),
        ('{"priceHint": 2}', 0.00005),
        (None, 0.00005),
        ([{"priceHint": 2}], 0.00005),
        ({"priceHint": "nonsense"}, 0.00005),
        ({"priceHint": math.nan}, 0.00005),
        ({"priceHint": math.inf}, 0.00005),
        ({"priceHint": 2.5}, 0.00005),
        ({"priceHint": 99}, 0.00005),
    ],
)
def test_price_tolerance_defensively_falls_back(metadata, expected):
    assert price_tolerance(metadata) == expected


class _LazyMetadata(Mapping):
    """Representative of yfinance 1.7's dict-like HistoryMetadata wrapper."""

    def __getitem__(self, key):
        return {"priceHint": 2, "exchangeName": "NMS"}[key]

    def __iter__(self):
        return iter(("priceHint", "exchangeName"))

    def __len__(self):
        return 2


def test_json_safe_preserves_provider_mapping_instead_of_stringifying_it():
    assert json_safe(_LazyMetadata()) == {"priceHint": 2, "exchangeName": "NMS"}


def test_compare_symbol_survives_exact_live_string_metadata_path(monkeypatch):
    frame = pd.DataFrame(
        {"open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0],
         "adj_close": [11.0], "volume": [100.0]},
        index=pd.to_datetime(["2026-09-23"]),
    )

    def fetch(symbol, interval, lookback_days):
        # Reproduces the value that reached price_tolerance in the failed run.
        return frame.copy(), "{'priceHint': 2, 'exchangeName': 'NMS'}", {}

    monkeypatch.setattr("diagnostics.compare_native_derived_eod._fetch_native", fetch)
    result = compare_symbol("AAPL")

    assert all(entry["status"] == "ok" for entry in result["intervals"].values())
    assert all(entry["price_tolerance"] == 0.00005 for entry in result["intervals"].values())
    assert all("metadata type str" in entry["price_tolerance_source"]
               for entry in result["intervals"].values())
    assert result["intervals"]["1d"]["provider_metadata"].startswith("{'priceHint'")


def test_main_persists_startup_and_failure_evidence_before_nonzero_exit(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected provider state")

    monkeypatch.setattr("diagnostics.compare_native_derived_eod.compare_symbol", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_native_derived_eod.py", "--symbols", "AAPL", "--observations", "1",
         "--output-dir", str(tmp_path)],
    )

    with pytest.raises(RuntimeError, match="unexpected provider state"):
        main()

    raw = (tmp_path / "raw_native_observations.json").read_text()
    assert "unexpected provider state" in raw
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "provider_metadata.json").exists()


def test_payload_hash_and_symbol_validation():
    frame = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-09-21"]))
    assert payload_hash(frame) == payload_hash(frame.copy())
    changed = frame.copy()
    changed.iloc[0, 0] = 2.0
    assert payload_hash(frame) != payload_hash(changed)
    assert parse_symbols("aapl, MSFT BRK-B") == ["AAPL", "MSFT", "BRK-B"]
