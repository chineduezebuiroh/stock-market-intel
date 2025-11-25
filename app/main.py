import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

st.set_page_config(page_title='Market Intel', layout='wide')
st.title('Market Intel — Signals')

#tabs = st.tabs(['Daily','Weekly','Monthly','Quarterly','130m','Fut Hourly','Fut 4H','Fut Daily'])
tabs = st.tabs([
    'Daily',
    'Weekly',
    'Monthly',
    'Quarterly',
    '130m',
    'Fut Hourly',
    'Fut 4H',
    'Fut Daily',
    'Stocks C: D/M Shortlist',
])

"""
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
"""
mapping = {
    'Daily': 'snapshot_stocks_daily.parquet',
    'Weekly': 'snapshot_stocks_weekly.parquet',
    'Monthly': 'snapshot_stocks_monthly.parquet',
    'Quarterly': 'snapshot_stocks_quarterly.parquet',
    '130m': 'snapshot_stocks_intraday_130m.parquet',
    'Fut Hourly': 'snapshot_futures_hourly.parquet',
    'Fut 4H': 'snapshot_futures_four_hour.parquet',
    'Fut Daily': 'snapshot_futures_daily.parquet',
    'Stocks C: D/M Shortlist': 'snapshot_combo_stocks_stocks_c_dm_shortlist.parquet',
}

"""
for tab in mapping:
    with tabs[list(mapping.keys()).index(tab)]:
        p = DATA / mapping[tab]
        if p.exists():
            df = pd.read_parquet(p)
            st.dataframe(df.reset_index(drop=True))
        else:
            st.info('No snapshot yet.')
"""

for name, filename in mapping.items():
    idx = list(mapping.keys()).index(name)
    with tabs[idx]:
        p = DATA / filename
        if not p.exists():
            st.info('No snapshot yet.')
            continue

        df = pd.read_parquet(p).reset_index(drop=True)

        if name == 'Stocks C: D/M Shortlist':
            st.subheader("Stocks C — Daily / Monthly (Shortlist)")
            if 'signal' in df.columns:
                long_df = df[df['signal'] == 'long']
                short_df = df[df['signal'] == 'short']
                watch_df = df[df['signal'] == 'watch']

                st.markdown("**Long Setups**")
                if long_df.empty:
                    st.write("No long setups.")
                else:
                    st.dataframe(long_df)

                st.markdown("**Short Setups**")
                if short_df.empty:
                    st.write("No short setups.")
                else:
                    st.dataframe(short_df)

                st.markdown("**Watchlist**")
                if watch_df.empty:
                    st.write("No watchlist names.")
                else:
                    st.dataframe(watch_df)
            else:
                st.dataframe(df)
        else:
            st.dataframe(df)
