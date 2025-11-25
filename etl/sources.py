import os
import pandas as pd
import yfinance as yf

ALPHA_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')

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
        df5 = df5.tz_localize(None)

    out = resample_ohlcv(df5, '130T')
    return out
