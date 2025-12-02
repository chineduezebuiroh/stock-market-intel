from __future__ import annotations

# etl/sources.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

ALPHA_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')


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




def load_intraday_yf(ticker: str, interval: str = '5m', period: str = '30d', session: str = 'regular') -> pd.DataFrame:
    prepost = (session == 'extended')
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        prepost=prepost,
        progress=False
    )
    if df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    df = df.rename(columns=str.lower)
    df = _sanitize_eod_df(df) # <-- New line added
    return df #[['open', 'high', 'low', 'close', 'volume']]

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ['open', 'high', 'low', 'close', 'volume']
    return out.dropna()


"""
def load_130m_from_5m(symbol: str, session: str = "regular") -> pd.DataFrame:
    
    #Build 130-minute bars from 5-minute Yahoo data.

    #Invariants:
      #- index tz-aware UTC from Yahoo -> converted to America/New_York -> tz-naive
      #- columns: open, high, low, close, adj_close, volume
      #- NO MultiIndex columns
    
    import yfinance as yf

    df = yf.download(
        symbol,
        interval="5m",
        period="60d",       # or whatever window you prefer
        progress=False,
        auto_adjust=False,  # avoid future default-change warning
        group_by="column",  # helps keep columns flat for single ticker
    )

    # If nothing came back, bail early
    if df is None or df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Flatten any MultiIndex columns (yfinance can return ('Open',) etc.)
    # ------------------------------------------------------------------
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        # For a single symbol, level 0 is usually 'Open','High',...
        cols = cols.get_level_values(0)

    df.columns = [str(c).lower().replace(" ", "_") for c in cols]

    # ------------------------------------------------------------------
    # Timezone handling + session filter
    # ------------------------------------------------------------------
    df = df.tz_convert("America/New_York")

    if session == "regular":
        # Regular trading hours
        df = df.between_time("09:30", "16:00")

    # ------------------------------------------------------------------
    # Resample 5m -> 130m
    # ------------------------------------------------------------------
    rule = "130min"

    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()

    # Adj close: if present, resample; otherwise mirror close
    if "adj_close" in df.columns:
        ac = df["adj_close"].resample(rule).last()
    else:
        ac = c

    out = pd.concat([o, h, l, c, ac, v], axis=1)
    out.columns = ["open", "high", "low", "close", "adj_close", "volume"]

    # Drop rows that are completely empty (e.g., incomplete first bucket)
    out = out.dropna(how="all")

    # Normalize index to tz-naive NY time
    out.index = out.index.tz_localize(None)
    out.index.name = "date"

    # Ensure plain string columns
    out.columns = out.columns.astype(str)

    return out
"""


def load_130m_from_5m(symbol: str, session: str = "regular") -> pd.DataFrame:
    """
    Build 130-minute bars from 5-minute Yahoo data.

    Invariants:
      - index tz-aware UTC from Yahoo -> converted to America/New_York -> tz-naive
      - columns: open, high, low, close, adj_close, volume
      - NO MultiIndex columns
      - 130m bars anchored at 09:30 America/New_York
    """
    import yfinance as yf

    df = yf.download(
        symbol,
        interval="5m",
        period="60d",
        progress=False,
        auto_adjust=False,  # keep close + adj close separate
        group_by="column",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Flatten any MultiIndex columns
    # ------------------------------------------------------------------
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        cols = cols.get_level_values(0)

    df.columns = [str(c).lower().replace(" ", "_") for c in cols]

    # ------------------------------------------------------------------
    # Timezone handling + session filter
    # ------------------------------------------------------------------
    df = df.tz_convert("America/New_York")

    if session == "regular":
        df = df.between_time("09:30", "16:00")

    if df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Resample 5m -> 130m with clean 09:30 anchoring
    # ------------------------------------------------------------------
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    has_adj = "adj_close" in df.columns
    if has_adj:
        agg["adj_close"] = "last"

    resampled = df.resample(
        "130min",
        origin="start_day",
        offset="9h30min",
        label="left",
        closed="left",
    ).agg(agg)

    if resampled.empty:
        return pd.DataFrame()

    # If no adj_close from Yahoo, mirror close
    if not has_adj:
        resampled["adj_close"] = resampled["close"]

    # Enforce column order and drop any rows that are all-NaN
    resampled = resampled[["open", "high", "low", "close", "adj_close", "volume"]]
    resampled = resampled.dropna(how="all")

    # Normalize index to tz-naive NY time
    resampled.index = resampled.index.tz_localize(None)
    resampled.index.name = "date"

    # Ensure plain string columns
    resampled.columns = resampled.columns.astype(str)

    return resampled



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

    out = df[cols].resample("YE").agg(agg)  # year-end frequency
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
