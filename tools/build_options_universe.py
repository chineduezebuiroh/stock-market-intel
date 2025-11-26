import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import yfinance as yf

# ---------- CONFIGURABLE CONSTANTS ----------

# Minimum market cap in USD (e.g., 2_000_000_000 = $2B)
MIN_MARKET_CAP = 1_000_000_000

# Sleep between requests (seconds) to be gentle on Yahoo
REQUEST_SLEEP_SECONDS = 0.15

# Maximum symbols to process (set to None for no cap; useful while testing)
MAX_SYMBOLS = None  # e.g., 500 for a quick run

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "ref"
REF_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = REF_DIR / "options_eligible.csv"


# ---------- CANDIDATE UNIVERSE HELPERS ----------
"""
def get_candidate_symbols() -> List[str]:
    candidates = set()

    try:
        sp500 = yf.tickers_sp500()
        print(f"[INFO] Loaded {len(sp500)} S&P 500 tickers from yfinance.")
        candidates.update(sp500)
    except Exception as e:
        print(f"[WARN] Could not load S&P 500 tickers: {e}")

    try:
        nasdaq = yf.tickers_nasdaq()
        print(f"[INFO] Loaded {len(nasdaq)} NASDAQ tickers from yfinance.")
        candidates.update(nasdaq)
    except Exception as e:
        print(f"[WARN] Could not load NASDAQ tickers: {e}")

    try:
        nyse = yf.tickers_nyse()
        print(f"[INFO] Loaded {len(nyse)} NYSE tickers from yfinance.")
        candidates.update(nyse)
    except Exception as e:
        print(f"[WARN] Could not load NYSE tickers: {e}")



    # You can add more sources here later if needed
    # e.g., yf.tickers_other() for misc indices

    candidates = sorted(candidates)
    if MAX_SYMBOLS is not None:
        candidates = candidates[:MAX_SYMBOLS]

    print(f"[INFO] Using {len(candidates)} candidate symbols in total.")
    return candidates
"""





def get_candidate_symbols() -> List[str]:
    """
    Build an initial candidate universe of stock tickers.

    This version uses the official NASDAQ Trader symbol directories:
      - NASDAQ-listed: nasdaqlisted.txt
      - Other-listed (NYSE, AMEX, etc.): otherlisted.txt

    Later, we can add more sources or local CSVs if needed.
    """
    import pandas as pd

    candidates = set()

    # NASDAQ-listed
    try:
        nasdaq_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        nasdaq_df = pd.read_csv(
            nasdaq_url,
            sep="|",
            dtype=str,
            engine="python",
            skipfooter=1,  # last line is "File Creation Time"
        )
        nasdaq_syms = nasdaq_df["Symbol"].dropna().astype(str).str.strip().str.upper().tolist()
        candidates.update(nasdaq_syms)
        print(f"[INFO] Loaded {len(nasdaq_syms)} NASDAQ-listed tickers.")
    except Exception as e:
        print(f"[WARN] Could not load NASDAQ-listed tickers: {e}")

    # Other-listed (includes NYSE, AMEX, etc.)
    try:
        other_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        other_df = pd.read_csv(
            other_url,
            sep="|",
            dtype=str,
            engine="python",
            skipfooter=1,  # last line is "File Creation Time"
        )
        # 'ACT Symbol' is the trading symbol in this file
        col = "ACT Symbol" if "ACT Symbol" in other_df.columns else "SYMBOL"
        other_syms = other_df[col].dropna().astype(str).str.strip().str.upper().tolist()
        candidates.update(other_syms)
        print(f"[INFO] Loaded {len(other_syms)} other-listed (NYSE/AMEX/etc.) tickers.")
    except Exception as e:
        print(f"[WARN] Could not load other-listed tickers: {e}")

    candidates = sorted(candidates)
    if MAX_SYMBOLS is not None:
        candidates = candidates[:MAX_SYMBOLS]

    print(f"[INFO] Using {len(candidates)} candidate symbols in total.")
    return candidates



# ---------- METADATA FETCHING ----------

def fetch_symbol_metadata(sym: str) -> Dict[str, Any]:
    """
    Fetch metadata for a single symbol via yfinance.

    Returns a dict with keys:
      symbol, has_options, market_cap, name, sector, industry, sub_industry
    """
    tkr = yf.Ticker(sym)

    # Check if the symbol has options
    try:
        opt_chain = tkr.options
        has_options = bool(opt_chain and len(opt_chain) > 0)
    except Exception:
        has_options = False

    # Fetch core info
    try:
        info = tkr.info  # or tkr.get_info() in newer yfinance
    except Exception:
        info = {}

    market_cap = info.get("marketCap")
    name = info.get("longName") or info.get("shortName") or ""
    sector = info.get("sector") or ""
    industry = info.get("industry") or ""

    # Placeholder for sub_industry; can be refined later
    sub_industry = ""

    return {
        "symbol": sym,
        "has_options": has_options,
        "market_cap": market_cap,
        "name": name,
        "sector": sector,
        "industry": industry,
        "sub_industry": sub_industry,
    }


# ---------- MAIN BUILD LOGIC ----------

def build_options_universe():
    """
    Build options_eligible.csv with:
      - one row per symbol
      - only symbols that:
          * have listed options (via yfinance)
          * meet MIN_MARKET_CAP
      - include metadata for later joins (sector/industry/sub_industry)
    """
    symbols = get_candidate_symbols()
    rows: List[Dict[str, Any]] = []

    for i, sym in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] Fetching {sym} ...", end="", flush=True)
        try:
            meta = fetch_symbol_metadata(sym)
        except Exception as e:
            print(f" ERROR ({e})")
            continue

        # Basic filters
        if not meta["has_options"]:
            print(" skip (no options)")
            time.sleep(REQUEST_SLEEP_SECONDS)
            continue

        mc = meta["market_cap"]
        if mc is None or mc < MIN_MARKET_CAP:
            print(" skip (market cap below threshold)")
            time.sleep(REQUEST_SLEEP_SECONDS)
            continue

        rows.append(meta)
        print(" ok")
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not rows:
        print("[WARN] No symbols passed filters. Not writing options_eligible.csv.")
        return

    df = pd.DataFrame(rows)
    df = df[["symbol", "name", "market_cap", "sector", "industry", "sub_industry"]]

    df.to_csv(OUT_PATH, index=False)
    print(f"[OK] Wrote {len(df)} symbols to {OUT_PATH}")


if __name__ == "__main__":
    build_options_universe()
