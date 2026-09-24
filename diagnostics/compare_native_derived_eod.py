"""Read-only comparison of current Yahoo native and daily-derived EOD bars."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns={name: name.lower().replace(" ", "_") for name in FIELDS})
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("America/New_York").tz_localize(None)
    return out[
        [
            name.lower().replace(" ", "_")
            for name in FIELDS
            if name.lower().replace(" ", "_") in out
        ]
    ]


def derive_terminal(daily: pd.DataFrame, interval: str) -> pd.Series:
    terminal = daily.index.max()
    if interval == "1wk":
        start = terminal - pd.Timedelta(days=terminal.weekday())
    elif interval == "1mo":
        start = terminal.replace(day=1)
    else:
        raise ValueError(interval)
    current = daily.loc[
        (daily.index.normalize() >= start.normalize()) & (daily.index <= terminal)
    ]
    return pd.Series(
        {
            "open": current.open.iloc[0],
            "high": current.high.max(),
            "low": current.low.min(),
            "close": current.close.iloc[-1],
            "adj_close": current.adj_close.iloc[-1],
            "volume": current.volume.sum(),
        }
    )


def payload_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_json(date_format="iso", orient="split").encode()
    ).hexdigest()


def compare_symbol(symbol: str, *, lookback: str = "6mo") -> dict:
    fetch_id = f"{symbol}-{datetime.now(timezone.utc).isoformat()}"
    frames, metadata, timings = {}, {}, {}
    ticker = yf.Ticker(symbol)
    for interval in ("1d", "1wk", "1mo"):
        started = datetime.now(timezone.utc).isoformat()
        raw = ticker.history(
            period=lookback, interval=interval, auto_adjust=False, actions=False
        )
        completed = datetime.now(timezone.utc).isoformat()
        frames[interval] = canonicalize(raw)
        timings[interval] = {
            "fetch_started_at": started,
            "fetch_completed_at": completed,
        }
        try:
            metadata[interval] = ticker.get_history_metadata() or {}
        except Exception as exc:
            metadata[interval] = {"metadata_error": str(exc)}

    daily = frames["1d"]
    result = {
        "symbol": symbol,
        "fetch_id": fetch_id,
        "provider": "Yahoo via yfinance",
        "request": {
            "period": lookback,
            "auto_adjust": False,
            "actions": False,
            "prepost": False,
        },
        "last_daily_market_date": daily.index.max().date().isoformat(),
        "intervals": {},
    }
    for interval in ("1d", "1wk", "1mo"):
        frame = frames[interval]
        entry = {
            **timings[interval],
            "interval_label": frame.index.max().isoformat(),
            "terminal": {
                k: None if pd.isna(v) else float(v) for k, v in frame.iloc[-1].items()
            },
            "payload_sha256": payload_hash(frame),
            "provider_metadata": metadata[interval],
        }
        if interval != "1d":
            derived = derive_terminal(daily, interval)
            entry["daily_derived"] = {k: float(v) for k, v in derived.items()}
            entry["native_minus_derived"] = {}
            for key in derived.index:
                native = entry["terminal"].get(key)
                entry["native_minus_derived"][key] = (
                    float(native - derived[key])
                    if native is not None and np.isfinite(native)
                    else None
                )
        result["intervals"][interval] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "SPY"])
    parser.add_argument("--lookback", default="6mo")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/native-derived-eod.json")
    )
    args = parser.parse_args()
    report = {
        "evidence_kind": "current_direct_provider_observation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": [
            compare_symbol(symbol, lookback=args.lookback) for symbol in args.symbols
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
