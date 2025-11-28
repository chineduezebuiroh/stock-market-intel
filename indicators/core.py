import numpy as np
import pandas as pd

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def wema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0):
    basis = sma(close, length)
    dev = close.rolling(length, min_periods=length).std()
    upper = basis + mult * dev
    lower = basis - mult * dev
    return basis, upper, lower

def slope(series: pd.Series, length: int) -> pd.Series:
    return series.diff(length) / length

def apply_core(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    close = df['close']
    high, low = df['high'], df['low']
    vol = df['volume']

    ema_fast = int(params.get('ema_fast', 8))
    ema_slow = int(params.get('ema_slow', 21))
    wild_len = int(params.get('wild_len', 14))
    rsi_len = int(params.get('rsi_len', 14))
    vol_sma = int(params.get('volume_sma', 20))

    df[f'sma_{ema_fast}'] = sma(close, ema_fast)
    df[f'ema_{ema_fast}'] = ema(close, ema_fast)
    df[f'ema_{ema_slow}'] = ema(close, ema_slow)
    df[f'wema_{wild_len}'] = wema(close, wild_len)
    #df[f'rsi_{rsi_len}'] = rsi_wilder(close, rsi_len)

    df[f'ema_{ema_fast}_slope'] = slope(df[f'ema_{ema_fast}'], 3)
    df[f'ema_{ema_slow}_slope'] = slope(df[f'ema_{ema_slow}'], 3)
    df[f'rsi_{rsi_len}_slope'] = slope(df[f'rsi_{rsi_len}'], 3)

    df[f'volume_sma_{vol_sma}'] = sma(vol, vol_sma)
    df[f'atr_{wild_len}'] = atr_wilder(high, low, close, wild_len)
    return df
