import streamlit as st
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
