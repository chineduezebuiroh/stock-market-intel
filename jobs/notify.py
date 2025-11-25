import os
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
