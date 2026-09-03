"""Read-only abstraction for the rolling per-symbol daily OHLC store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class PriceHistory:
    symbol: str
    source_key: str
    frame: pd.DataFrame


class RollingDailyPriceSource:
    """Load ``bars/stocks_daily/<SYMBOL>.parquet`` via an injected reader.

    The injected reader keeps this module incapable of storage writes and lets a
    caller select local parquet, S3 GetObject, or a pinned S3 object version.
    Values are normalized but never adjusted or rewritten.
    """

    def __init__(self, reader: Callable[[str], pd.DataFrame], prefix: str = "") -> None:
        self._reader = reader
        self._prefix = prefix.strip("/")

    def key(self, symbol: str) -> str:
        relative = f"bars/stocks_daily/{symbol.strip().upper()}.parquet"
        return f"{self._prefix}/{relative}" if self._prefix else relative

    def load(self, symbol: str) -> PriceHistory:
        key = self.key(symbol)
        frame = self._reader(key).copy()
        if frame.empty:
            return PriceHistory(symbol.upper(), key, frame)
        if not isinstance(frame.index, pd.DatetimeIndex):
            date_column = next(
                (name for name in ("date", "datetime", "timestamp") if name in frame),
                None,
            )
            if date_column is None:
                raise ValueError(f"{key} has no datetime index or date column")
            frame = frame.set_index(
                pd.to_datetime(frame.pop(date_column), errors="coerce")
            )
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert("America/New_York").tz_localize(None)
        frame.index = frame.index.normalize()
        required = {"high", "low", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{key} missing outcome columns: {sorted(missing)}")
        frame = frame.loc[frame.index.notna()].sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        return PriceHistory(symbol.upper(), key, frame)


class CachedPriceSource:
    """Memoize a price source so an audit loads every symbol at most once."""

    def __init__(self, source: RollingDailyPriceSource) -> None:
        self._source = source
        self._cache: dict[str, PriceHistory] = {}

    def load(self, symbol: str) -> PriceHistory:
        normalized = symbol.strip().upper()
        if normalized not in self._cache:
            self._cache[normalized] = self._source.load(normalized)
        return self._cache[normalized]


def local_parquet_reader(root: Path) -> Callable[[str], pd.DataFrame]:
    return lambda key: pd.read_parquet(root / key)
