#!/usr/bin/env python3
"""Read-only provider reconstruction of stock state at a historical close.

This diagnostic deliberately does not use the production storage abstraction or
any production job entry point. It downloads daily Yahoo observations ending immediately
after the requested market date, applies a strict local-date cutoff, constructs
partial weekly/monthly bars, and invokes the configured production indicators.

The result is a present-day reconstruction, not an immutable copy of what Yahoo
returned to production in the past.  Provider corrections and the production
weekly/monthly interval responses cannot be recovered by this method.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf

from diagnostics.investigate_scoring_lineage import recompute_indicators

OHLCV = ("open", "high", "low", "close", "adj_close", "volume")
MATERIAL_INDICATORS = (
    "ma_trend_bullish",
    "ma_trend_bearish",
    "exh_abs_pa_current_bar",
    "exh_abs_pa_prior_bar",
    "macdv_core_bull",
    "macdv_core_bear",
    "ttm_squeeze_pro",
    "sig_vol_current_bar",
    "wyckoff_stage",
)


def normalize_provider_daily(frame: pd.DataFrame, market_date: date) -> pd.DataFrame:
    """Normalize Yahoo daily data and reject every post-cutoff observation."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(OHLCV))
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(column).lower().replace(" ", "_") for column in out.columns]
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_convert("America/New_York").tz_localize(None)
    out = out.loc[out.index.notna()]
    out = out.loc[pd.Series(out.index.date <= market_date, index=out.index)]
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    if "adj_close" not in out:
        out["adj_close"] = out.get("close", np.nan)
    for column in OHLCV:
        if column not in out:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[list(OHLCV)]


def aggregate_partial_bars(daily: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Build start-labelled bars, retaining the current partial week/month."""
    if timeframe == "daily":
        return daily.copy()
    periods = {
        "weekly": daily.index.to_period("W-SUN").start_time,
        "monthly": daily.index.to_period("M").start_time,
    }
    if timeframe not in periods:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    grouped = daily.groupby(periods[timeframe], sort=True)
    result = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        adj_close=("adj_close", "last"),
        volume=("volume", lambda values: values.sum(min_count=1)),
    )
    result.index.name = daily.index.name or "date"
    return result


def fetch_yahoo_daily(symbol: str, start: date, end_exclusive: date) -> pd.DataFrame:
    """Fetch unadjusted daily Yahoo observations without actions."""
    return yf.Ticker(symbol).history(
        interval="1d",
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=False,
        actions=False,
    )


def reconstruct(
    symbol: str,
    market_date: date,
    indicator_config: Path,
    *,
    fetcher: Callable[[str, date, date], pd.DataFrame] = fetch_yahoo_daily,
    years_of_history: int = 8,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """Return canonical bars, material final values, and provenance metadata."""
    start = market_date - timedelta(days=366 * years_of_history)
    end_exclusive = market_date + timedelta(days=1)
    raw = fetcher(symbol, start, end_exclusive)
    daily = normalize_provider_daily(raw, market_date)
    if daily.empty:
        raise RuntimeError("provider returned no eligible observations")

    bars = {
        timeframe: aggregate_partial_bars(daily, timeframe)
        for timeframe in ("daily", "weekly", "monthly")
    }
    rows: list[dict[str, Any]] = []
    errors: dict[str, dict[str, str]] = {}
    for timeframe, frame in bars.items():
        values, timeframe_errors = recompute_indicators(
            frame, timeframe, indicator_config
        )
        errors[timeframe] = timeframe_errors
        last = frame.iloc[-1]
        row: dict[str, Any] = {
            "timeframe": timeframe,
            "timestamp": str(frame.index[-1]),
            "observation_count": len(frame),
            **{column: last[column] for column in OHLCV},
        }
        row.update(
            {
                indicator: values.get(indicator, np.nan)
                for indicator in MATERIAL_INDICATORS
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    metadata = {
        "symbol": symbol.upper(),
        "market_date": market_date.isoformat(),
        "provider": "Yahoo via yfinance",
        "requested_start": start.isoformat(),
        "requested_end_exclusive": end_exclusive.isoformat(),
        "eligible_daily_observations": len(daily),
        "first_daily_observation": str(daily.index.min()),
        "last_daily_observation": str(daily.index.max()),
        "maximum_input_market_date": daily.index.max().date().isoformat(),
        "post_cutoff_observations": int((daily.index.date > market_date).sum()),
        "exact_historical_production_evidence": False,
        "limitations": (
            "Present-day Yahoo daily data may include revisions. Production fetched "
            "1d/1wk/1mo independently; reconstructed weekly/monthly bars are aggregated "
            "from daily observations and cannot recover past provider payloads."
        ),
        "indicator_errors": errors,
    }
    return bars, summary, metadata


def compare_persisted_row(
    summary: pd.DataFrame, persisted: pd.Series | None
) -> pd.DataFrame:
    """Compare reconstruction with an immutable combo row when supplied."""
    roles = {"daily": "lower", "weekly": "middle", "monthly": "upper"}
    records: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        timeframe = row["timeframe"]
        role = roles[timeframe]
        for field in (*OHLCV, *MATERIAL_INDICATORS):
            column = f"{role}_{field}"
            available = persisted is not None and column in persisted.index
            stored = persisted[column] if available else np.nan
            reconstructed = row[field]
            if not available:
                status = "unavailable"
            elif pd.isna(stored) and pd.isna(reconstructed):
                status = "match"
            elif pd.isna(stored) or pd.isna(reconstructed):
                status = "mismatch"
            elif isinstance(stored, (int, float, np.number)) and isinstance(
                reconstructed, (int, float, np.number)
            ):
                status = "match" if np.isclose(stored, reconstructed) else "mismatch"
            else:
                status = "match" if stored == reconstructed else "mismatch"
            records.append(
                {
                    "timeframe": timeframe,
                    "role": role,
                    "field": field,
                    "reconstructed": reconstructed,
                    "persisted_column": column,
                    "persisted": stored,
                    "comparison": status,
                }
            )
    return pd.DataFrame(records)


def _persisted_row(path: Path | None, symbol: str) -> pd.Series | None:
    if path is None:
        return None
    frame = pd.read_parquet(path)
    if "symbol" not in frame:
        raise ValueError("persisted combo artifact has no symbol column")
    rows = frame.loc[frame["symbol"].astype(str).str.upper().eq(symbol.upper())]
    if rows.empty:
        raise ValueError(f"{symbol} is absent from persisted combo artifact")
    return rows.iloc[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="CRWD")
    parser.add_argument("--market-date", default="2026-09-21")
    parser.add_argument("--indicator-config", default="config/indicator_params.yaml")
    parser.add_argument("--persisted-combo", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("forensic_artifacts"))
    args = parser.parse_args()

    market_date = date.fromisoformat(args.market_date)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        bars, summary, metadata = reconstruct(
            args.symbol.upper(), market_date, Path(args.indicator_config)
        )
    except Exception as exc:
        metadata = {
            "symbol": args.symbol.upper(),
            "market_date": market_date.isoformat(),
            "provider": "Yahoo via yfinance",
            "status": "unavailable",
            "exact_historical_production_evidence": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (output / "reconstruction_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(json.dumps(metadata, indent=2))
        return
    for timeframe, frame in bars.items():
        frame.to_csv(output / f"{args.symbol.lower()}_{timeframe}_reconstructed.csv")
    summary.to_csv(output / "reconstructed_material_state.csv", index=False)
    persisted = _persisted_row(args.persisted_combo, args.symbol)
    compare_persisted_row(summary, persisted).to_csv(
        output / "reconstructed_vs_persisted.csv", index=False
    )
    (output / "reconstruction_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
