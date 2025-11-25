import pathlib
import textwrap

def write_file(root: pathlib.Path, rel_path: str, content: str):
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")

def main():
    root = pathlib.Path(".").resolve()

    files = {
        "README.md": """            # Stock Market Intel

        Free-tier stock & futures screening with Streamlit + Parquet + GitHub Actions.

        ## Quick Start

        1. Create and activate a virtual environment:

           ```bash
           python -m venv .venv
           # macOS/Linux
           source .venv/bin/activate
           # Windows (PowerShell)
           .venv\\Scripts\\Activate
           ```

        2. Install dependencies:

           ```bash
           pip install --upgrade pip
           pip install -r requirements.txt
           ```

        3. Run a quick smoke test:

           ```bash
           python jobs/run_timeframe.py stocks daily --cascade
           python jobs/run_timeframe.py stocks intraday_130m --cascade
           python jobs/run_timeframe.py futures hourly --cascade
           python jobs/run_timeframe.py futures four_hour --cascade
           ```

        4. Launch the app:

           ```bash
           streamlit run app/main.py
           ```
        """,

        "requirements.txt": """            pandas
        numpy
        yfinance
        pyarrow
        fastparquet
        duckdb
        PyYAML
        streamlit
        requests
        """,

        "setup.sh": """            #!/usr/bin/env bash
        set -euo pipefail
        python -m venv .venv
        if [ -f ".venv/bin/activate" ]; then
          source .venv/bin/activate
        else
          source .venv/Scripts/activate
        fi
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        echo "✅ Environment ready. Activate it with: source .venv/bin/activate"
        """,

        ".gitignore": """            .venv/
        __pycache__/
        .DS_Store
        data/
        */__pycache__/
        .streamlit/
        """,

        "config/timeframes.yaml": """            stocks:
          intraday_130m:
            session: regular
            window_bars: 300
            universe: shortlist_stocks
          daily:
            session: regular
            window_bars: 300
            universe: options_eligible
          weekly:
            session: regular
            window_bars: 300
            universe: options_eligible
          monthly:
            session: regular
            window_bars: 250
            universe: options_eligible
          quarterly:
            session: regular
            window_bars: 250
            universe: options_eligible

        futures:
          hourly:
            session: extended
            window_bars: 250
            universe: shortlist_futures
          four_hour:
            session: extended
            window_bars: 300
            universe: shortlist_futures
          daily:
            session: extended
            window_bars: 300
            universe: shortlist_futures
          weekly:
            session: extended
            window_bars: 250
            universe: shortlist_futures
          monthly:
            session: extended
            window_bars: 250
            universe: shortlist_futures
        """,

        "config/shortlists.yaml": """            shortlist_stocks:
          - AAPL
          - AMZN
          - MSFT
          - GOOGL
          - META
          - NVDA
          - TSLA
          - V
          - SPY
          - QQQ

        shortlist_futures:
          - ES=F   # /ES → ES=F (S&P 500 E-mini)
          - NQ=F   # /NQ → NQ=F (Nasdaq 100 E-mini)
          - YM=F   # /YM → YM=F (Dow E-mini)
          - GC=F   # /GC → GC=F (Gold)
          - PL=F   # /PL → PL=F (Platinum)
          - SI=F   # /SI → SI=F (Silver)
          - ZS=F   # /ZS → ZS=F (Soybeans)
          - ZL=F   # /ZL → ZL=F (Soybean Oil)
          - ZM=F   # /ZM → ZM=F (Soybean Meal)
        """,

        "config/screens/daily.yaml": """            name: "Daily — Momentum Pullback (template)"
        params:
          rsi_len: 14
          ema_fast: 8
          ema_slow: 21
          wild_len: 14
          volume_sma: 20
        rules:
          - expr: "(ema_8 > wema_14 and sma_8 > wema_14) or (ema_8 > sma_8)"
          - expr: "rsi_14 between 45 and 65"
          - expr: "volume_sma_20 > 500000"
          - expr: "market_cap > 2e9"
        sort:
          - "-rsi_14_slope"
          - "-ema_8_slope"
        limit: 50
        """,

        "config/screens/weekly.yaml": """            name: "Weekly — Trend Continuation (template)"
        params:
          rsi_len: 14
          ema_fast: 13
          ema_slow: 34
          wild_len: 14
        rules:
          - expr: "(ema_13 > wema_14 and sma_13 > wema_14) or (ema_13 > sma_13)"
          - expr: "ema_13 > ema_34"
          - expr: "rsi_14 between 50 and 70"
        limit: 50
        """,

        "config/screens/intraday_130m.yaml": """            name: "130m — MTF Aligned (template)"
        params:
          ema_fast: 21
          ema_slow: 55
          wild_len: 14
        rules:
          - expr: "(ema_21 > wema_14 and sma_21 > wema_14) or (ema_21 > sma_21)"
          - expr: "ema_21 > ema_55"
        limit: 30
        """,

        "app/main.py": """            import streamlit as st
        import pandas as pd
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[1]
        DATA = ROOT / 'data'

        st.set_page_config(page_title='Market Intel', layout='wide')
        st.title('Market Intel — Signals')

        tabs = st.tabs(['Daily','Weekly','Monthly','Quarterly','130m','Fut Hourly','Fut 4H','Fut Daily'])

        mapping = {
            'Daily': 'snapshot_stocks_daily.parquet',
            'Weekly': 'snapshot_stocks_weekly.parquet',
            'Monthly': 'snapshot_stocks_monthly.parquet',
            'Quarterly': 'snapshot_stocks_quarterly.parquet',
            '130m': 'snapshot_stocks_intraday_130m.parquet',
            'Fut Hourly': 'snapshot_futures_hourly.parquet',
            'Fut 4H': 'snapshot_futures_four_hour.parquet',
            'Fut Daily': 'snapshot_futures_daily.parquet',
        }

        for tab in mapping:
            with tabs[list(mapping.keys()).index(tab)]:
                p = DATA / mapping[tab]
                if p.exists():
                    df = pd.read_parquet(p)
                    st.dataframe(df.reset_index(drop=True))
                else:
                    st.info('No snapshot yet.')
        """,

        "etl/sources.py": """            import os
        import pandas as pd
        import yfinance as yf

        ALPHA_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')

        def load_eod(ticker: str, start: str = '2000-01-01', end: str | None = None, session: str = 'regular') -> pd.DataFrame:
            prepost = (session == 'extended')
            df = yf.download(ticker, start=start, end=end, interval='1d', auto_adjust=False, prepost=prepost, progress=False)
            if df.empty:
                return pd.DataFrame(columns=['open','high','low','close','volume'])
            df = df.rename(columns=str.lower)
            return df[['open','high','low','close','volume']]

        def load_intraday_yf(ticker: str, interval: str = '5m', period: str = '30d', session: str = 'regular') -> pd.DataFrame:
            prepost = (session == 'extended')
            df = yf.download(ticker, period=period, interval=interval, prepost=prepost, progress=False)
            if df.empty:
                return pd.DataFrame(columns=['open','high','low','close','volume'])
            df = df.rename(columns=str.lower)
            return df[['open','high','low','close','volume']]

        def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
            o = df['open'].resample(rule).first()
            h = df['high'].resample(rule).max()
            l = df['low'].resample(rule).min()
            c = df['close'].resample(rule).last()
            v = df['volume'].resample(rule).sum()
            out = pd.concat([o,h,l,c,v], axis=1)
            out.columns = ['open','high','low','close','volume']
            return out.dropna()

        def load_130m_from_5m(ticker: str, session: str = 'regular') -> pd.DataFrame:
            df5 = load_intraday_yf(ticker, interval='5m', period='30d', session=session)
            if df5.empty:
                return df5
            if getattr(df5.index, 'tz', None) is not None:
                df5 = df5.tz_localize(None)
            out = resample_ohlcv(df5, '130T')
            return out
        """,

        "etl/window.py": """            from pathlib import Path
        import pandas as pd

        def parquet_path(root: Path, timeframe: str, ticker: str) -> Path:
            return root / f"timeframe={timeframe}" / f"ticker={ticker}" / "data.parquet"

        def update_fixed_window(df_new: pd.DataFrame, existing: pd.DataFrame, window_bars: int) -> pd.DataFrame:
            df = pd.concat([existing, df_new]).sort_index()
            df = df[~df.index.duplicated(keep='last')]
            if len(df) > window_bars:
                df = df.iloc[-window_bars:]
            return df
        """,

        "indicators/core.py": """            import numpy as np
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
            df[f'rsi_{rsi_len}'] = rsi_wilder(close, rsi_len)

            df[f'ema_{ema_fast}_slope'] = slope(df[f'ema_{ema_fast}'], 3)
            df[f'ema_{ema_slow}_slope'] = slope(df[f'ema_{ema_slow}'], 3)
            df[f'rsi_{rsi_len}_slope'] = slope(df[f'rsi_{rsi_len}'], 3)

            df[f'volume_sma_{vol_sma}'] = sma(vol, vol_sma)
            df[f'atr_{wild_len}'] = atr_wilder(high, low, close, wild_len)
            return df
        """,

        "screens/engine.py": """            import pandas as pd
        from typing import Dict

        def eval_rule(df: pd.DataFrame, expr: str) -> pd.Series:
            expr = expr.replace(' between ', ' _between_ ')
            if '_between_' in expr:
                left, rest = expr.split('_between_')
                lo_str, hi_str = rest.split('and')
                lo = float(lo_str.strip())
                hi = float(hi_str.strip())
                return (df[left.strip()] >= lo) & (df[left.strip()] <= hi)
            return pd.eval(expr, engine='python', parser='pandas', local_dict={c: df[c] for c in df.columns})

        def run_screen(df: pd.DataFrame, yaml_cfg: Dict) -> pd.DataFrame:
            rules = yaml_cfg.get('rules', [])
            out = df.copy()
            for r in rules:
                m = eval_rule(out, r['expr'])
                out = out[m]
            for key in yaml_cfg.get('sort', []):
                ascending = not key.startswith('-')
                col = key.lstrip('+-')
                if col in out.columns:
                    out = out.sort_values(col, ascending=ascending)
            lim = yaml_cfg.get('limit')
            if lim:
                out = out.head(int(lim))
            return out
        """,

        "jobs/run_timeframe.py": """            import sys
        from pathlib import Path
        import pandas as pd
        import yaml

        from etl.sources import load_eod, load_130m_from_5m
        from etl.window import parquet_path, update_fixed_window
        from indicators.core import apply_core
        from screens.engine import run_screen

        ROOT = Path(__file__).resolve().parents[1]
        DATA = ROOT / 'data'
        CFG = ROOT / 'config'

        with open(CFG / 'timeframes.yaml','r') as f:
            TF_CFG = yaml.safe_load(f)
        with open(CFG / 'shortlists.yaml','r') as f:
            SHORT = yaml.safe_load(f)

        def symbols_for(universe: str):
            if universe == 'options_eligible':
                p = ROOT / 'ref' / 'options_eligible.csv'
                return pd.read_csv(p)['symbol'].tolist() if p.exists() else []
            return SHORT.get(universe, [])

        def ingest_one(namespace: str, timeframe: str, symbols, session: str, window_bars: int):
            for sym in symbols:
                if namespace == 'stocks' and timeframe == 'intraday_130m':
                    df_new = load_130m_from_5m(sym, session=session)
                else:
                    df_new = load_eod(sym, start='2000-01-01', end=None, session=session)
                parquet = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
                parquet.parent.mkdir(parents=True, exist_ok=True)
                existing = pd.read_parquet(parquet) if parquet.exists() else pd.DataFrame()
                merged = update_fixed_window(df_new, existing, window_bars)
                merged = apply_core(merged, params={})
                merged.to_parquet(parquet)

        def run(namespace: str, timeframe: str, cascade: bool = False):
            cfg = TF_CFG[namespace][timeframe]
            session = cfg['session']
            window_bars = int(cfg['window_bars'])
            symbols = symbols_for(cfg['universe'])

            ingest_one(namespace, timeframe, symbols, session, window_bars)

            if cascade:
                if namespace == 'stocks':
                    if timeframe == 'intraday_130m':
                        for tf in ['daily','weekly']:
                            c = TF_CFG['stocks'][tf]
                            ingest_one('stocks', tf, symbols, c['session'], int(c['window_bars']))
                    elif timeframe == 'daily':
                        for tf in ['weekly','monthly']:
                            c = TF_CFG['stocks'][tf]
                            ingest_one('stocks', tf, symbols, c['session'], int(c['window_bars']))
                elif namespace == 'futures':
                    if timeframe == 'hourly':
                        c = TF_CFG['futures']['daily']
                        ingest_one('futures', 'daily', symbols, c['session'], int(c['window_bars']))
                    elif timeframe == 'four_hour':
                        c = TF_CFG['futures']['weekly']
                        ingest_one('futures', 'weekly', symbols, c['session'], int(c['window_bars']))
                    elif timeframe == 'daily':
                        c = TF_CFG['futures']['monthly']
                        ingest_one('futures', 'monthly', symbols, c['session'], int(c['window_bars']))

            screen_path = CFG / 'screens' / f'{timeframe}.yaml'
            if screen_path.exists():
                with open(screen_path,'r') as f:
                    screen_cfg = yaml.safe_load(f)
                rows = []
                for sym in symbols:
                    p = parquet_path(DATA, f"{namespace}_{timeframe}", sym)
                    if not p.exists():
                        continue
                    df = pd.read_parquet(p)
                    if df.empty:
                        continue
                    last = df.iloc[-1:].copy()
                    last['symbol'] = sym
                    rows.append(last)
                if rows:
                    snap = pd.concat(rows)
                    snap = run_screen(snap, screen_cfg)
                    out = ROOT / 'data' / f'snapshot_{namespace}_{timeframe}.parquet'
                    snap.to_parquet(out)

        if __name__ == '__main__':
            ns = sys.argv[1]
            tf = sys.argv[2]
            cascade = ('--cascade' in sys.argv)
            run(ns, tf, cascade=cascade)
        """,

        "jobs/notify.py": """            import os
        import pandas as pd
        import requests
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[1]
        DATA = ROOT / 'data'

        TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

        def diff_symbols(current: pd.DataFrame, previous: pd.DataFrame):
            cur = set(current['symbol']) if current is not None else set()
            prev = set(previous['symbol']) if previous is not None else set()
            return sorted(cur - prev), sorted(prev - cur)

        def send_telegram(text: str):
            if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
                return
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            try:
                requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
            except Exception as e:
                print(f"[telegram] send error: {e}")

        def main(snapshot_name: str):
            cur_path = DATA / snapshot_name
            prev_path = DATA / (snapshot_name.replace('.parquet','_prev.parquet'))
            cur = pd.read_parquet(cur_path) if cur_path.exists() else None
            prev = pd.read_parquet(prev_path) if prev_path.exists() else None
            if cur is None or cur.empty:
                return
            added, removed = diff_symbols(cur, prev)
            if added or removed:
                msg = f"{snapshot_name}: NEW {added} | EXIT {removed}"
                print(msg)
                send_telegram(msg)
            if cur is not None:
                cur.to_parquet(prev_path)

        if __name__ == '__main__':
            main('snapshot_stocks_daily.parquet')
        """,
    }

    for rel, content in files.items():
        write_file(root, rel, content)

    print("✅ Scaffold rebuilt. Next steps:")
    print("1) python -m venv .venv")
    print("2) source .venv/bin/activate  (or .venv\\Scripts\\Activate on Windows)")
    print("3) pip install -r requirements.txt")
    print("4) streamlit run app/main.py")

if __name__ == "__main__":
    main()
