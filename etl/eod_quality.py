"""Data-quality gates for canonical stock end-of-day bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_STOCK_EOD_FIELDS = ("open", "high", "low", "close", "volume")


class StockEODDataQualityError(ValueError):
    """Raised when stock EOD data cannot safely be used for indicators."""


def valid_stock_eod_rows(df: pd.DataFrame) -> pd.Series:
    """Return a mask for rows with finite OHLCV (volume may legitimately be zero)."""
    if df is None:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=df.index, dtype=bool)
    for field in REQUIRED_STOCK_EOD_FIELDS:
        if field not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        values = pd.to_numeric(df[field], errors="coerce")
        mask &= np.isfinite(values)
        if field == "volume":
            mask &= values >= 0
    return mask


def require_valid_terminal_stock_eod(df: pd.DataFrame, *, context: str) -> None:
    """Fail closed unless the terminal canonical stock EOD row has valid OHLCV."""
    if df is None or df.empty:
        raise StockEODDataQualityError(f"{context}: no canonical rows")
    if not bool(valid_stock_eod_rows(df).iloc[-1]):
        invalid = [
            field
            for field in REQUIRED_STOCK_EOD_FIELDS
            if field not in df.columns
            or not np.isfinite(
                pd.to_numeric(
                    pd.Series([df.iloc[-1].get(field)]), errors="coerce"
                ).iloc[0]
            )
        ]
        raise StockEODDataQualityError(
            f"{context}: terminal row {df.index[-1]!s} has invalid required fields {invalid}"
        )


@dataclass(frozen=True)
class StockEODMergeResult:
    frame: pd.DataFrame
    retained_existing_timestamps: tuple[pd.Timestamp, ...]
    dropped_malformed_timestamps: tuple[pd.Timestamp, ...]


def merge_stock_eod_window(
    df_new: pd.DataFrame, existing: pd.DataFrame, window_bars: int
) -> StockEODMergeResult:
    """Merge without allowing malformed provider rows to replace valid observations.

    Whole malformed rows are discarded; fields from independently observed rows are
    never coalesced. A malformed terminal refresh is recoverable only when a valid
    existing row has the exact same canonical timestamp.
    """
    from etl.window import update_fixed_window

    if df_new is None or df_new.empty:
        raise StockEODDataQualityError("provider stock EOD payload: no canonical rows")

    valid_new_mask = valid_stock_eod_rows(df_new)
    bad = df_new.index[~valid_new_mask]
    terminal = df_new.index[-1]
    existing_valid = valid_stock_eod_rows(existing)
    can_retain_terminal = (
        terminal in existing.index and bool(existing_valid.loc[terminal])
        if existing is not None and not existing.empty and terminal in existing.index
        else False
    )
    if terminal in bad and not can_retain_terminal:
        require_valid_terminal_stock_eod(df_new, context="provider stock EOD payload")

    clean_new = df_new.loc[valid_new_mask]
    merged = update_fixed_window(clean_new, existing, window_bars)
    require_valid_terminal_stock_eod(merged, context="merged stock EOD window")
    retained = (
        (pd.Timestamp(terminal),) if terminal in bad and can_retain_terminal else ()
    )
    return StockEODMergeResult(
        frame=merged,
        retained_existing_timestamps=retained,
        dropped_malformed_timestamps=tuple(pd.Timestamp(ts) for ts in bad),
    )
