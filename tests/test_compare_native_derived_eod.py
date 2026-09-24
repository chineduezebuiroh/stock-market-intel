import pandas as pd

from diagnostics.compare_native_derived_eod import derive_terminal, payload_hash


def test_derive_partial_week_and_month_through_latest_daily_session():
    index = pd.to_datetime(["2026-09-18", "2026-09-21", "2026-09-22"])
    daily = pd.DataFrame(
        {
            "open": [90, 100, 103],
            "high": [99, 105, 108],
            "low": [89, 98, 102],
            "close": [98, 104, 107],
            "adj_close": [98, 104, 107],
            "volume": [10, 20, 30],
        },
        index=index,
    )
    weekly = derive_terminal(daily, "1wk")
    monthly = derive_terminal(daily, "1mo")
    assert weekly.to_dict() == {
        "open": 100,
        "high": 108,
        "low": 98,
        "close": 107,
        "adj_close": 107,
        "volume": 50,
    }
    assert monthly.to_dict() == {
        "open": 90,
        "high": 108,
        "low": 89,
        "close": 107,
        "adj_close": 107,
        "volume": 60,
    }


def test_payload_hash_is_stable_and_content_sensitive():
    frame = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-09-21"]))
    assert payload_hash(frame) == payload_hash(frame.copy())
    changed = frame.copy()
    changed.iloc[0, 0] = 2.0
    assert payload_hash(frame) != payload_hash(changed)
