import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

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


"""
def load_eod(ticker: str, start: str = '2000-01-01', end: str | None = None, session: str = 'regular') -> pd.DataFrame:
    prepost = (session == 'extended')
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval='1d',
        auto_adjust=False,
        prepost=prepost,
        progress=False
    )
    if df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    df = df.rename(columns=str.lower)
    return df[['open', 'high', 'low', 'close', 'volume']]


import yfinance as yf
import pandas as pd
"""

def load_eod(symbol: str, timeframe: str, window_bars: int, session: str) -> pd.DataFrame:
    """
    Load EOD-style bars using yfinance for the given timeframe and window size.

    We:
      - compute a start date based on timeframe + window_bars,
      - call yfinance with a timeout and raise_errors=False,
      - rename columns to a standard OHLCV schema,
      - return a DataFrame (possibly empty).
    """
    ticker = yf.Ticker(symbol)

    # Map our logical timeframe to yfinance interval
    if timeframe == "daily":
        interval = "1d"
    elif timeframe == "weekly":
        interval = "1wk"
    elif timeframe in ("monthly", "quarterly", "yearly"):
        interval = "1mo"  # we'll resample/trim via update_fixed_window if needed
    else:
        interval = "1d"  # sensible default

    start = compute_start_date_for_window(timeframe, window_bars)

    try:
        df = ticker.history(
            start=start,
            end=None,
            interval=interval,
            auto_adjust=False,
            actions=False,
            timeout=10,        # key: don't hang forever
            raise_errors=False,
        )
    except Exception as e:
        print(f"[WARN] load_eod: hard failure for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        print(f"[WARN] load_eod: empty DataFrame for {symbol} ({timeframe})")
        return pd.DataFrame()

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    
    # 🔹 normalize index to EST
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)    
    
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
