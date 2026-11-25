import warnings

import pandas as pd
import pyarrow.parquet as pq
import pytest

from jobs import run_timeframe


BASE_COLS = ["open", "high", "low", "close", "volume", "indicator"]


def _bars():
    return {
        "AAA": pd.DataFrame(
            [[1.0, 3.0, 0.5, 2.0, 100, None],
             [2.0, 4.0, 1.5, 3.0, 200, 1.25]],
            columns=BASE_COLS, index=pd.to_datetime(["2026-11-16", "2026-11-17"]),
        ),
        # The last row is invalid only for intraday futures; the prior row is used.
        "BBB": pd.DataFrame(
            [[3.0, 5.0, 2.5, 4.0, 300, None],
             [4.0, 6.0, 3.5, None, 400, 2.5]],
            columns=BASE_COLS, index=pd.to_datetime(["2026-11-16", "2026-11-18"]),
        ),
    }


def _legacy_snapshot(frames, namespace, timeframe):
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for symbol, frame in frames.items():
            if namespace == "futures" and timeframe in ("intraday_1h", "intraday_4h"):
                frame = frame.dropna(subset=["open", "high", "low", "close"], how="any")
            row = frame.iloc[-1][BASE_COLS].copy()
            row["symbol"] = symbol
            rows.append(row)
    result = pd.DataFrame(rows)[BASE_COLS + ["symbol"]]
    if result.index.name is None:
        result.index.name = "date"
    result.columns = result.columns.astype(str)
    return result


@pytest.mark.parametrize("namespace,timeframe", [("stocks", "intraday_4h"), ("futures", "intraday_1h"), ("futures", "intraday_4h")])
def test_snapshot_parity_and_parquet_schema(namespace, timeframe, monkeypatch, tmp_path):
    frames = _bars()
    monkeypatch.setattr(run_timeframe, "get_snapshot_base_cols", lambda *args: BASE_COLS)
    monkeypatch.setattr(run_timeframe.storage, "exists", lambda path: path.stem in frames)
    monkeypatch.setattr(run_timeframe.storage, "load_parquet", lambda path: frames[path.stem])
    expected = _legacy_snapshot(frames, namespace, timeframe)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        actual = run_timeframe.build_timeframe_snapshot_frame(namespace, timeframe, ["AAA", "BBB", "MISSING"])
    assert not any("incompatible dtype" in str(w.message) for w in captured)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
    assert all(pd.api.types.is_numeric_dtype(actual[col]) for col in BASE_COLS)
    assert list(actual.columns) == BASE_COLS + ["symbol"]
    old_path, new_path = tmp_path / "old.parquet", tmp_path / "new.parquet"
    expected.to_parquet(old_path)
    actual.to_parquet(new_path)
    assert pq.read_schema(old_path).equals(pq.read_schema(new_path), check_metadata=True)
    pd.testing.assert_frame_equal(pd.read_parquet(new_path), pd.read_parquet(old_path))


def test_numeric_series_symbol_assignment_is_deprecated_when_supported():
    warning = getattr(pd.errors, "Pandas4Warning", None)
    if warning is None:
        pytest.skip("Pandas4Warning is emitted by newer pandas")
    row = pd.Series([1.0, 2.0], index=["open", "close"])
    with pytest.warns(warning, match="incompatible dtype"):
        row["symbol"] = "AAA"
