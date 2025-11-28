# etl/sources.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from __future__ import annotations

ALPHA_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')


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
        return "1y", window_bars * 2 * 365

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

    end_utc = pd.Timestamp.utcnow().tz_localize("UTC")
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
    return df[['open', 'high', 'low', 'close', 'volume']]

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ['open', 'high', 'low', 'close', 'volume']
    return out.dropna()

def load_130m_from_5m(ticker: str, session: str = 'regular') -> pd.DataFrame:
    df5 = load_intraday_yf(ticker, interval='5m', period='30d', session=session)
    if df5.empty:
        return df5
    
    # Remove timezone if present
    if getattr(df5.index, 'tz', None) is not None:
        #df5 = df5.tz_localize(None)
        df5.index = df5.index.tz_convert("America/New_York").tz_localize(None)
        

    out = resample_ohlcv(df5, '130T')
    return out
