from __future__ import annotations
# etl/sources.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import signal
from contextlib import contextmanager

#from etl.session import ensure_current_hour_stub_1h
from etl.futures_partial import (
    _try_load_recent_intraday,
    patch_partial_1h_from_5m,
    patch_partial_4h_from_5m,
)

from typing import Any


class TimeoutException(Exception):
    pass
    
ALPHA_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')

INTRADAY_REPAIR_TELEMETRY: dict[str, dict[str, Any]] = {}

# ======================================================
# --------------- Helper Function(s) ---------------
# ======================================================
@contextmanager
def timeout(seconds: int, msg: str = "Timeout"):
    def handler(signum, frame):
        raise TimeoutException(msg)
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


"""
def _repair_intraday_bad_ticks(
    df: pd.DataFrame,
    *,
    max_bar_range_pct: float = 0.08,
) -> pd.DataFrame:
"""
def _repair_intraday_bad_ticks(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str = "stocks_intraday_4h",
    source: str = "60m",
    max_bar_range_pct: float = 0.08,
) -> pd.DataFrame:
    """
    Repair obviously bad intraday OHLC rows instead of dropping them.

    If a row has absurd OHLC behavior, replace OHLC with prior valid close
    and set volume=0. This preserves continuous bar structure for indicators.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return out

    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)

    bad = (
        out[required].isna().any(axis=1)
        | (out[required] <= 0).any(axis=1)
        | (out["high"] < out["low"])
        | (((out["high"] - out["low"]) / out["close"].abs()) > max_bar_range_pct)
    )

    if not bad.any():
        return out

    n_bad = int(bad.sum())
    if n_bad:
        key = f"{symbol or 'UNKNOWN'}|{source}"
        INTRADAY_REPAIR_TELEMETRY[key] = {
            "symbol": symbol or "",
            "timeframe": timeframe,
            "source": source,
            "bars_total": int(len(out)),
            "bars_repaired": n_bad,
        }
        print(f"[CLEAN] {symbol or ''} repairing {n_bad} bad intraday bars ({source})", flush=True)

    #print(f"[CLEAN] repairing {int(bad.sum())} bad intraday bars", flush=True)

    prior_close = out["close"].where(~bad).ffill()

    for idx in out.index[bad]:
        px = prior_close.loc[idx]
        if pd.isna(px):
            continue

        out.loc[idx, "open"] = px
        out.loc[idx, "high"] = px
        out.loc[idx, "low"] = px
        out.loc[idx, "close"] = px

        if "adj_close" in out.columns:
            out.loc[idx, "adj_close"] = px

        if "volume" in out.columns:
            out.loc[idx, "volume"] = 0.0

    return out


def _ensure_continuous_4h_5pm_anchor(
    df_4h: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure continuous 4h bars on the 17:00 ET anchor grid:
      17, 21, 01, 05, 09, 13

    Missing buckets are filled with carry-forward close and volume=0.
    """
    if df_4h is None or df_4h.empty:
        return df_4h

    out = df_4h.copy().sort_index()
    out.index = pd.to_datetime(out.index)

    start = out.index.min()
    end = out.index.max()

    full_idx = pd.date_range(start=start, end=end, freq="4h")
    full_idx = pd.DatetimeIndex([ts for ts in full_idx if _is_stock_extended_4h_bucket(ts)])

    out = out.reindex(full_idx)
    out.index.name = df_4h.index.name or "date"

    # Carry-forward close is the synthetic price anchor
    prior_close = out["close"].ffill()

    missing_price = out[["open", "high", "low", "close"]].isna().all(axis=1)

    for c in ["open", "high", "low", "close"]:
        out.loc[missing_price, c] = prior_close.loc[missing_price]

    if "adj_close" in out.columns:
        out.loc[missing_price, "adj_close"] = prior_close.loc[missing_price]
    else:
        out["adj_close"] = out["close"]

    if "volume" in out.columns:
        out.loc[missing_price, "volume"] = 0.0
        out["volume"] = out["volume"].fillna(0.0)
    else:
        out["volume"] = 0.0

    # If the first row was missing and had no prior close, drop it.
    out = out.dropna(subset=["open", "high", "low", "close"], how="any")

    return out


def _is_stock_extended_4h_bucket(ts: pd.Timestamp) -> bool:
    """
    TOS-style stock extended-hours 4h bucket calendar.

    4h anchor grid: 17, 21, 01, 05, 09, 13

    Observed pattern:
      - Mon-Thu: 01, 05, 09, 13, 17, 21
      - Fri:     01, 05, 09, 13, 17
      - Sat:     none
      - Sun:     17, 21

    Holidays are intentionally not handled yet.
    """
    ts = pd.Timestamp(ts)
    wd = ts.weekday()  # Mon=0 ... Sun=6
    hr = ts.hour

    if wd in (0, 1, 2, 3):  # Mon-Thu
        return hr in (1, 5, 9, 13, 17, 21)

    if wd == 4:  # Friday
        return hr in (1, 5, 9, 13, 17)

    if wd == 5:  # Saturday
        return False

    if wd == 6:  # Sunday
        return hr in (17, 21)

    return False
    
# ======================================================
# ---------------- Actual Processes ----------------
# ======================================================
def _agg_ohlcv_intraday(g: pd.DataFrame) -> pd.Series:
    # g is all rows that floor to the same hour
    out = {
        "open": g["open"].iloc[0],
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].iloc[-1],
        "volume": g["volume"].sum(),
    }
    if "adj_close" in g.columns:
        out["adj_close"] = g["adj_close"].iloc[-1]
    return pd.Series(out)


def normalize_intraday_1h_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    # 1) Force to hour-start grid
    out.index = out.index.floor("h")

    # 2) Merge duplicates that now share the same hour
    if out.index.has_duplicates:
        out = (
            out.sort_index()
               .groupby(level=0, sort=True)
               .apply(_agg_ohlcv_intraday)
        )
        out.index.name = "Datetime"

    return out


def _sanitize_eod_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final safety pass for load_eod:
    - Flatten any MultiIndex columns.
    - Normalize column names to our canonical set.
    - Ensure we only keep the OHLCV fields we care about.
    """
    if df is None or df.empty:
        return df

    # 1) If yfinance returns MultiIndex columns (e.g. ('Open','AAPL')),
    #    collapse to the *first* level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2) Normalize column names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "Adj close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # 3) Keep only canonical columns (drop anything weird)
    keep = ["open", "high", "low", "close", "adj_close", "volume"]
    cols = [c for c in keep if c in df.columns]

    # If adj_close isn't present, that's fine; we just skip it.
    df = df[cols].copy()

    # 4) Make absolutely sure columns are plain strings
    df.columns = df.columns.astype(str)

    return df


def compute_start_date_for_window(timeframe: str, window_bars: int) -> str:
    """
    Compute a reasonable start date so we only fetch enough history to
    cover the desired rolling window (plus margin), instead of 25+ years.

    Very rough approximations (can be tuned):
      - daily: 1.5x window bars in calendar days
      - weekly: 7x window bars in weeks
      - monthly/quarterly/yearly: 19,000 (~50 years of data)
    """
    today = datetime.utcnow().date()

    if timeframe == "daily":
        days_back = int(window_bars * 1.5)
    elif timeframe == "weekly":
        days_back = int(window_bars * 7)
    elif timeframe in ("monthly", "quarterly", "yearly"):
        days_back = 19000 #ignore int(window_bars * 3 * 30), 50 years x 12 months x 31 days = 18600 days, round to 19000
    else:
        # default fallback: window_bars = number of days
        days_back = int(window_bars)

    start_date = today - timedelta(days=days_back)
    return start_date.isoformat()


def _timeframe_to_interval_and_lookback(timeframe: str, window_bars: int) -> tuple[str, int]:
    """
    Map internal timeframe names to yfinance intervals and a rough lookback in days.
    We over-fetch a bit (x2) so we have breathing room for missing days / holidays.
    """
    tf = timeframe.lower()

    if tf == "daily":
        return "1d", window_bars * 2
    if tf == "weekly":
        # 1wk bars; ~7 days each
        return "1wk", window_bars * 2 * 7
    if tf == "monthly":
        # 1mo bars; ~30 days each
        return "1mo", window_bars * 2 * 30
    if tf == "quarterly":
        # 3mo bars; ~90 days each
        return "3mo", window_bars * 2 * 90
    if tf == "yearly":
        # 1y bars; ~365 days each
        return "3mo", window_bars * 8 * 365

    # Fallback: treat unknown as daily
    return "1d", window_bars * 2


def safe_load_eod(
    symbol: str,
    timeframe: str = "daily",
    window_bars: int = 300,
    session: str = "regular",
    timeout_sec: int = 30,
):
    """
    Wraps load_eod() with a global timeout to prevent hangs.
    Signature mirrors load_eod so we don't rely on kwargs.
    """
    try:
        with timeout(timeout_sec, msg=f"EOD fetch timed out for {symbol}"):
            # IMPORTANT: match how you normally call load_eod
            return load_eod(symbol, timeframe=timeframe, window_bars=window_bars, session=session)
            
    except TimeoutException as e:
        print(f"[TIMEOUT] {e}")
        return None
    except Exception as e:
        print(f"[ERROR] safe_load_eod({symbol}): {e}")
        return None


def load_eod(
    symbol: str,
    timeframe: str = "daily",
    window_bars: int = 300,
    session: str = "regular",
) -> pd.DataFrame:
    """
    Load end-of-day (or higher timeframe) bars for a single symbol using yfinance.

    Guarantees:
      - one symbol per DataFrame (no per-symbol columns)
      - flat columns: open, high, low, close, adj_close, volume
      - index: tz-naive datetime in America/New_York
      - at most `window_bars` rows (sliding window)

    `session` is kept for parity with intraday loaders; for EOD bars yfinance
    already gives you RTH-style bars, so we don't currently filter further.
    """
    interval, lookback_days = _timeframe_to_interval_and_lookback(timeframe, window_bars)
    
    end_utc = pd.Timestamp.utcnow()
    if end_utc.tzinfo is None:
        end_utc = end_utc.tz_localize("UTC")
    else:
        end_utc = end_utc.tz_convert("UTC")

    start_utc = end_utc - timedelta(days=lookback_days)

    t = yf.Ticker(symbol)

    try:
        df = t.history(
            interval=interval,
            start=start_utc,
            end=end_utc,
            auto_adjust=False,
            actions=False,
        )
    except Exception as e:
        # If yfinance blows up for any reason, just return empty
        print(f"[WARN] load_eod failed for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1) Deal with any MultiIndex columns (future-proofing)
    # ------------------------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        # Typical yfinance MultiIndex: level 0 = field, level 1 = symbol
        # Try to pick the slice for our symbol if present.
        try:
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, axis=1, level=-1)
        except Exception:
            # Fallback: flatten into strings; we'll tidy up below.
            df.columns = [
                "_".join(str(x) for x in level if x not in (None, ""))
                for level in df.columns.values
            ]

    # ------------------------------------------------------------------
    # 2) Normalize index: America/New_York, tz-naive
    # ------------------------------------------------------------------
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    # ------------------------------------------------------------------
    # 3) Normalize column names to our canonical set
    # ------------------------------------------------------------------
    rename_map: dict[str, str] = {}
    for col in df.columns:
        lc = str(col).lower()

        if lc.startswith("open"):
            rename_map[col] = "open"
        elif lc.startswith("high"):
            rename_map[col] = "high"
        elif lc.startswith("low"):
            rename_map[col] = "low"
        elif lc.startswith("close") and "adj" not in lc:
            rename_map[col] = "close"
        elif lc.startswith("adj close") or lc.startswith("adj_close") or "adjclose" in lc:
            rename_map[col] = "adj_close"
        elif lc.startswith("volume"):
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    # Keep only the columns we care about; add missing as NaN
    core_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in core_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[core_cols]

    # ------------------------------------------------------------------
    # 4) Clean index & enforce sliding window
    # ------------------------------------------------------------------
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if window_bars is not None and len(df) > window_bars:
        df = df.iloc[-window_bars:]

    df = _sanitize_eod_df(df)
    return df


def load_intraday_yf(
    ticker: str,
    interval: str = '5m',
    period: str = '30d',
    session: str = 'regular',
) -> pd.DataFrame:
    prepost = (session == 'extended')

    print(f"[YF] {ticker} interval={interval} period={period} prepost={prepost}", flush=True)
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        prepost=prepost,
        progress=False,
        auto_adjust=False,   # keep raw OHLCV, like load_eod
    )
    if df is None or df.empty:
        #return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        return pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"])

    # Normalize index to America/New_York tz-naive (same as load_eod)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    # Normalize column names, then sanitize
    df = df.rename(columns=str.lower)
    df = _sanitize_eod_df(df)

    # 🔹 Guarantee adj_close exists (important for snapshots + indicators)
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # Optional: enforce column order
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in cols if c in df.columns]]

    return df


def load_futures_intraday(
    symbol: str,
    timeframe: str,
    window_bars: int,
    session: str = "extended",
) -> pd.DataFrame:
    """
    Futures intraday loader for 1h and 4h bars, backed by yfinance.

    - intraday_1h: pulls 60m bars directly.
    - intraday_4h: pulls 60m bars, resamples to 4H.

    We over-fetch by ~2x the requested window and let update_fixed_window()
    enforce the final bar count.
    """
    tf = timeframe.lower()

    if tf not in ("intraday_1h", "intraday_4h"):
        raise ValueError(f"Unsupported futures intraday timeframe: {timeframe}")

    # Rough bars/day for futures 1h:
    bars_per_day_1h = 23  # 23h futures session accounting for 1 hour maintenance break
    if tf == "intraday_1h":
        bars_per_day = bars_per_day_1h
    else:  # intraday_4h
        bars_per_day = (bars_per_day_1h + 1) / 4.0  # ~6 bars per day

    # Over-fetch by 2x
    approx_days = max(5, int((window_bars / max(bars_per_day, 1)) * 2))
    
    # Yahoo intraday (60m) history limit is ~730 days — clamp requests
    MAX_YF_60M_DAYS = 729  # safer than 730
    if approx_days > MAX_YF_60M_DAYS:
        approx_days = MAX_YF_60M_DAYS
    
    period = f"{approx_days}d"

    print(f"[YF] {symbol} {timeframe} requesting 60m period={period}", flush=True)
    
    # 1) Load 60m bars
    df_1h = load_intraday_yf(
        symbol,
        interval="60m",
        period=period,
        session=session,
    )

    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
        
    #collapse multiple / random minute bars into a singular 1-hour bar
    df_1h = normalize_intraday_1h_index(df_1h)

    # NEW: patch in partial current hour from recent 5m/15m
    df_recent = _try_load_recent_intraday(load_intraday_yf, symbol, session=session)
    df_1h = patch_partial_1h_from_5m(df_1h=df_1h, df_5m=df_recent)

    if tf == "intraday_1h":
        return df_1h

    # 2) For intraday_4h: resample 1h → 4h
    df_4h = resample_futures_1h_to_4h_5pm_anchor(df_1h)
    
    # NEW: patch in partial current 4h bucket from recent 5m/15m
    df_4h = patch_partial_4h_from_5m(df_4h=df_4h, df_5m=df_recent)

    return df_4h


def resample_futures_1h_to_4h_5pm_anchor(
    df_1h: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    add_current_stub: bool = True,
) -> pd.DataFrame:
    """
    Resample 1h futures bars into 4h bars, anchored to 5pm NY session start.

    - Produces partial 4h bars as soon as there is >=1 underlying 1h bar in that bucket.
    - Optionally adds a "current bucket" stub (NaNs) if the current 4h bucket is missing entirely.
    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()

    df_1h = df_1h.sort_index().copy()

    # Ensure adj_close exists
    if "adj_close" not in df_1h.columns:
        df_1h["adj_close"] = df_1h["close"]

    # Use caller-provided 'now' or current time (assumes df is NY tz-naive)
    if now is None:
        now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    else:
        now = pd.Timestamp(now)
        if getattr(now, "tzinfo", None) is not None:
            now = now.tz_convert("America/New_York").tz_localize(None)

    # Shift index so that 17:00 becomes "00:00" from resample POV
    shifted = df_1h.copy()
    shifted.index = pd.to_datetime(shifted.index) - pd.Timedelta(hours=17)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "adj_close": "last",
        # critical: empty buckets must remain NaN, not 0
        "volume": lambda x: x.sum(min_count=1),
    }

    out = shifted.resample("4h").agg(agg)

    # Shift back so bars are labeled at true session times
    out.index = out.index + pd.Timedelta(hours=17)

    # drop buckets with no price data
    out = out.dropna(subset=["open", "high", "low", "close"], how="all")
    
    # optional: if you want to ensure "close" exists for indicators
    #out = out.dropna(subset=["close"])


    # column order
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    out = out[[c for c in cols if c in out.columns]]
    out.columns = out.columns.astype(str)
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ['open', 'high', 'low', 'close', 'volume']
    return out.dropna()


def load_130m_from_5m(symbol: str, session: str = "regular") -> pd.DataFrame:
    """
    Build 130-minute bars from 5-minute Yahoo data, anchored at 09:30 ET.

    Invariants:
      - Yahoo tz-aware UTC -> converted to America/New_York -> tz-naive
      - columns: open, high, low, close, adj_close, volume
      - NO MultiIndex columns
      - 130m buckets per trading day:
          09:30, 11:40, 13:50, ... (as far as there is data)
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np

    # ------------------------------------------------------------------
    # 1) Download raw 5m bars
    # ------------------------------------------------------------------
    df = yf.download(
        symbol,
        interval="5m",
        period="60d",
        progress=False,
        auto_adjust=False,   # keep close + adj_close separate
        group_by="column",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 2) Flatten any MultiIndex columns
    # ------------------------------------------------------------------
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        cols = cols.get_level_values(0)

    df.columns = [str(c).lower().replace(" ", "_") for c in cols]

    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep]

    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # ------------------------------------------------------------------
    # 3) Timezone + regular session filter
    # ------------------------------------------------------------------
    df = df.tz_convert("America/New_York")

    if session == "regular":
        # Only keep 09:30–16:00 local time
        df = df.between_time("09:30", "16:00")

    if df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 4) Build 130m bars explicitly, per day
    # ------------------------------------------------------------------
    frames = []
    tz = df.index.tz

    for day, day_df in df.groupby(df.index.date):
        day_df = day_df.sort_index()

        # Anchor at 09:30 local
        anchor = pd.Timestamp(day).tz_localize(tz) + pd.Timedelta(hours=9, minutes=30)

        minutes_since_open = (day_df.index - anchor).total_seconds() / 60.0

        # Only bars at/after 09:30
        mask = minutes_since_open >= 0
        if not mask.any():
            continue

        day_df = day_df[mask]
        minutes_since_open = minutes_since_open[mask]

        # 0,1,2,... bucket ids for each 130m chunk
        bucket = (minutes_since_open // 130).astype(int)

        agg = day_df.groupby(bucket).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "adj_close": "last",
                "volume": "sum",
            }
        )

        # Map bucket id -> timestamp = 09:30 + 130m * bucket
        new_index = [anchor + pd.Timedelta(minutes=130 * int(k)) for k in agg.index]
        agg.index = pd.DatetimeIndex(new_index, tz=tz)

        # Drop empty bars (no trades)
        agg = agg.replace({"volume": {0: np.nan}})
        agg = agg.dropna(subset=["close", "volume"], how="any")

        if not agg.empty:
            frames.append(agg)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames).sort_index()

    # ------------------------------------------------------------------
    # 5) Final normalization + safety filter
    # ------------------------------------------------------------------
    out.index = out.index.tz_localize(None)
    out.index.name = "date"
    out = out[["open", "high", "low", "close", "adj_close", "volume"]]
    out.columns = out.columns.astype(str)

    # HARD GUARANTEE: no zero-volume or all-NaN rows
    bad_mask = (out["volume"] <= 0) | out["close"].isna()
    if bad_mask.any():
        out = out[~bad_mask]

    return out


def load_stocks_intraday_4h_extended(
    symbol: str,
    window_bars: int,
    session: str = "extended",
) -> pd.DataFrame:
    """
    Build stock extended-hours 4h bars using the futures-style architecture:

      - Pull 60m Yahoo bars.
      - Normalize random/minute-drift timestamps to hour-start grid.
      - Patch current partial 1h bar from recent 5m/15m data.
      - Resample 1h -> 4h using 17:00 ET anchor:
            17:00, 21:00, 01:00, 05:00, 09:00, 13:00
      - Patch current partial 4h bucket from recent 5m/15m data.

    This intentionally uses extended session data and should produce up to
    6 four-hour bars per day, subject to Yahoo data availability.

    Notes:
      - This is for stocks, not futures.
      - Stocks extended-hours data may be sparse / low volume overnight.
      - Yahoo may not provide true 24h stock bars for all symbols.
    """
    # Stocks do not trade 24h like futures, but with prepost=True yfinance
    # provides extended-hours where available.
    if session != "extended":
        print(
            f"[WARN] load_stocks_intraday_4h_extended called with session={session!r}; "
            "forcing extended session.",
            flush=True,
        )
        session = "extended"

    # Rough: 6 bars/day for the 4h output.
    # We pull 60m bars and over-fetch.
    bars_per_day_4h = 6
    approx_days = max(5, int((window_bars / max(bars_per_day_4h, 1)) * 2))

    # Yahoo 60m limit is about 730 days. Stay conservative.
    MAX_YF_60M_DAYS = 729
    approx_days = min(approx_days, MAX_YF_60M_DAYS)

    period = f"{approx_days}d"

    print(f"[YF] {symbol} stocks intraday_4h requesting 60m period={period}", flush=True)

    # 1) Load 60m extended-hours bars
    df_1h = load_intraday_yf(symbol, interval="60m", period=period, session=session)

    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"])

    # 2) Normalize to hour-start grid and merge duplicate/random minute bars
    # This helper is futures-named, but it is generic OHLCV hourly normalization.
    df_1h = normalize_intraday_1h_index(df_1h)

    # New helper logic
    df_1h = _repair_intraday_bad_ticks(df_1h, symbol=symbol, source="60m") # <- max_bar_range_pct=0.08 already set as default

    # 3) Patch partial current 1h bar from recent 5m/15m
    df_recent = _try_load_recent_intraday(load_intraday_yf, symbol, session=session)
    df_recent = _repair_intraday_bad_ticks(df_recent, symbol=symbol, source="5m") # <- max_bar_range_pct=0.08 already set as default

    df_1h = patch_partial_1h_from_5m(df_1h=df_1h, df_5m=df_recent, symbol=symbol, session_gate=None)

    # 4) Resample 1h -> 4h with same 17:00 ET anchor as futures
    df_4h = resample_futures_1h_to_4h_5pm_anchor(df_1h)

    if df_4h is None or df_4h.empty:
        return pd.DataFrame()

    # New helper logic
    df_4h = _ensure_continuous_4h_5pm_anchor(df_4h)

    # 5) Patch partial current 4h bucket from recent 5m/15m
    df_4h = patch_partial_4h_from_5m(df_4h=df_4h, df_5m=df_recent, symbol=symbol, session_gate=None)

    # 6) Final cleanup
    if df_4h is None or df_4h.empty:
        return pd.DataFrame()

    df_4h = df_4h.sort_index().copy()
    df_4h.index = pd.to_datetime(df_4h.index)
    df_4h.index.name = "date"

    if "adj_close" not in df_4h.columns and "close" in df_4h.columns:
        df_4h["adj_close"] = df_4h["close"]

    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df_4h = df_4h[[c for c in cols if c in df_4h.columns]]
    df_4h.columns = df_4h.columns.astype(str)

    # Do not keep all-NaN OHLC rows.
    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df_4h.columns]
    if ohlc_cols:
        df_4h = df_4h[~df_4h[ohlc_cols].isna().all(axis=1)]

    return df_4h


def _resample_monthly_to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly OHLCV into true quarterly bars, indexed at
    the *beginning* of each quarter (e.g., 2025-01-01, 2025-04-01, ...).

    Assumes df has:
        open, high, low, close, adj_close, volume
    and a DatetimeIndex already normalized.
    """
    if df is None or df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "adj_close": "last",
        "volume": "sum",
    }

    # 🔹 QS = Quarter Start (e.g., 2025-01-01, 2025-04-01, ...)
    out = df[cols].resample("QS").agg(agg)

    out = out.dropna(how="all")
    out.index = out.index.normalize()
    out.columns = out.columns.astype(str)

    return out


def load_quarterly_from_monthly(
    symbol: str,
    window_bars: int,
    session: str = "regular",
) -> pd.DataFrame:
    """
    Build true quarterly bars by:
      1) Loading monthly OHLCV via load_eod(..., timeframe="monthly"),
      2) Resampling to quarterly OHLCV with _resample_monthly_to_quarterly.

    We ask for more monthly bars than the target quarterly window so we
    have enough history to aggregate.
    """
    monthly_window = window_bars * 4  # 4 months per quarter (very conservative)

    df_monthly = load_eod(
        symbol,
        timeframe="monthly",
        window_bars=monthly_window,
        session=session,
    )

    if df_monthly is None or df_monthly.empty:
        return df_monthly

    df_quarterly = _resample_monthly_to_quarterly(df_monthly)
    return df_quarterly


def _resample_monthly_to_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly OHLCV into true yearly bars.

    Assumes df has:
        open, high, low, close, adj_close, volume
    and a DatetimeIndex already normalized to NYC/naive as in load_eod.
    """
    if df is None or df.empty:
        return df

    # Make sure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # Only work with the expected OHLCV columns
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "adj_close": "last",
        "volume": "sum",
    }

    out = df[cols].resample("YS").agg(agg)  # year-start frequency
    # Drop all-NaN rows (just in case)
    out = out.dropna(how="all")

    # Normalize index to date-only (no time component)
    out.index = out.index.normalize()

    # Ensure flat string columns
    out.columns = out.columns.astype(str)

    return out


def load_yearly_from_monthly(
    symbol: str,
    window_bars: int,
    session: str = "regular",
):
    """
    Build true yearly bars by:
      1) Loading monthly OHLCV via load_eod(..., timeframe="monthly"),
      2) Resampling to yearly OHLCV with _resample_monthly_to_yearly.

    We ask for more monthly bars than the target yearly window so we
    have enough history to aggregate.
    """
    # Ask for more monthly bars than years we want.
    # Very conservative: 12 months per year of history.
    monthly_window = window_bars * 12

    df_monthly = load_eod(
        symbol,
        timeframe="monthly",
        window_bars=monthly_window,
        session=session,
    )

    if df_monthly is None or df_monthly.empty:
        return df_monthly

    df_yearly = _resample_monthly_to_yearly(df_monthly)
    return df_yearly
