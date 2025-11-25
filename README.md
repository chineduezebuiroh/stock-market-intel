    # Stock Market Intel

Free-tier stock & futures screening with Streamlit + Parquet + GitHub Actions.

## Quick Start

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate
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
