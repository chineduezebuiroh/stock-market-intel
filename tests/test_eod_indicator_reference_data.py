from __future__ import annotations

import pandas as pd
import pytest

from indicators import composite_spy_qqq_volume_ma_ratio as indicator


def reference_frame(scale: float) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-01", periods=300)
    return pd.DataFrame({"volume": [scale * (i + 1) for i in range(300)]}, index=index)


def test_preloaded_eod_references_preserve_fallback_formula_and_cache_key(monkeypatch):
    frames = {"SPY": reference_frame(1.0), "QQQ": reference_frame(2.0)}
    calls = []
    monkeypatch.setattr(
        indicator,
        "load_eod",
        lambda symbol, timeframe: calls.append((symbol, timeframe)) or frames[symbol].copy(),
    )
    indicator._spy_qqq_vol_ma_for_timeframe.cache_clear()
    fallback = indicator._spy_qqq_vol_ma_for_timeframe("daily", 21)
    again = indicator._spy_qqq_vol_ma_for_timeframe("daily", 21)
    assert len(calls) == 2
    pd.testing.assert_series_equal(fallback[0], again[0])

    seeded = {(symbol, "daily"): frame for symbol, frame in frames.items()}
    with indicator.preloaded_eod_references(seeded):
        preloaded = indicator._spy_qqq_vol_ma_for_timeframe("daily", 21)
        pd.testing.assert_series_equal(preloaded[0], fallback[0])
        pd.testing.assert_series_equal(preloaded[1], fallback[1])
        assert len(calls) == 2


def test_preloaded_context_fails_closed_without_required_reference(monkeypatch):
    monkeypatch.setattr(
        indicator, "load_eod",
        lambda *args, **kwargs: pytest.fail("provider fallback used"),
    )
    with indicator.preloaded_eod_references({("SPY", "daily"): reference_frame(1.0)}):
        with pytest.raises(RuntimeError, match="QQQ"):
            indicator._spy_qqq_vol_ma_for_timeframe("daily", 21)

