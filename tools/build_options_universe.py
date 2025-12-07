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

CFG = ROOT / "config"
EXCLUSIONS_FILE = CFG / "excluded_symbols.csv"

# ---------- CANDIDATE UNIVERSE HELPERS ----------
def load_symbol_exclusions() -> set[str]:
    if not EXCLUSIONS_FILE.exists():
        return set()

    try:
        df = pd.read_csv(EXCLUSIONS_FILE)
    except Exception as e:
        print(f"[WARN] Failed to read exclusions from {EXCLUSIONS_FILE}: {e}")
        return set()

    col = None
    for candidate in ("symbol", "Symbol", "ticker", "Ticker"):
        if candidate in df.columns:
            col = candidate
            break

    if col is None:
        print(
            f"[WARN] {EXCLUSIONS_FILE} has no 'symbol' or 'ticker' column; "
            "no exclusions will be applied."
        )
        return set()

    symbols = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    return set(symbols)


def get_candidate_symbols() -> List[str]:
    """
    Build an initial candidate universe of stock tickers.

    This version:
      - Tries to download NASDAQ Trader symbol directories from the official URLs.
      - If download succeeds, uses that data AND caches it under ref/.
      - If download fails, falls back to existing local files in ref/.
      - If both fail, that source is skipped.

    Sources:
      - NASDAQ-listed: nasdaqlisted.txt
      - Other-listed (NYSE, AMEX, etc.): otherlisted.txt
    """
    import pandas as pd

    candidates = set()

    def load_symbol_table(
        source_name: str,
        url: str,
        local_filename: str,
        symbol_col: str,
    ) -> List[str]:
        """
        Try to load a symbol table from URL, cache to ref/, then fall back to local file.
        Returns a list of symbols (uppercase, stripped).
        """
        local_path = REF_DIR / local_filename

        # First try URL
        try:
            print(f"[INFO] Loading {source_name} from URL: {url}")
            df = pd.read_csv(
                url,
                sep="|",
                dtype=str,
                engine="python",
                skipfooter=1,  # last line is "File Creation Time"
            )

            # Cache a copy locally (as a pipe-delimited file)
            try:
                df.to_csv(local_path, sep="|", index=False)
                print(f"[INFO] Cached {source_name} to {local_path}")
            except Exception as e_cache:
                print(f"[WARN] Could not cache {source_name} to {local_path}: {e_cache}")

        except Exception as e_url:
            print(f"[WARN] Could not load {source_name} from URL ({e_url}).")
            # Fallback to local file if it exists
            if local_path.exists():
                try:
                    print(f"[INFO] Falling back to local {source_name}: {local_path}")
                    df = pd.read_csv(
                        local_path,
                        sep="|",
                        dtype=str,
                        engine="python",
                        skipfooter=1,
                    )
                except Exception as e_local:
                    print(f"[WARN] Could not load local {source_name} from {local_path}: {e_local}")
                    return []
            else:
                print(f"[WARN] No local {source_name} file found at {local_path}. Skipping.")
                return []

        # Extract symbol column
        if symbol_col not in df.columns:
            print(f"[WARN] Column '{symbol_col}' not found in {source_name} table.")
            return []

        syms = (
            df[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        print(f"[INFO] Loaded {len(syms)} {source_name} tickers.")
        return syms

    # NASDAQ-listed
    nasdaq_syms = load_symbol_table(
        source_name="NASDAQ-listed",
        url="https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        local_filename="nasdaqlisted.txt",
        symbol_col="Symbol",
    )
    candidates.update(nasdaq_syms)

    # Other-listed (includes NYSE, AMEX, etc.)
    other_syms = load_symbol_table(
        source_name="other-listed (NYSE/AMEX/etc.)",
        url="https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        local_filename="otherlisted.txt",
        symbol_col="ACT Symbol",  # primary trading symbol column
    )
    candidates.update(other_syms)

    candidates = sorted(candidates)
    if MAX_SYMBOLS is not None:
        candidates = candidates[:MAX_SYMBOLS]

    print(f"[INFO] Using {len(candidates)} candidate symbols in total.")

    exclusions = load_symbol_exclusions()
    if exclusions:
        candidates["symbol_norm"] = (
            candidates["symbol"].astype(str).str.strip().str.upper()
        )
        before = len(candidates)
        candidates = candidates[~candidates["symbol_norm"].isin(exclusions)].copy()
        candidates = candidates.drop(columns=["symbol_norm"])
        after = len(candidates)
    
        print(
            f"[INFO] Excluded {before - after} symbols based on {EXCLUSIONS_FILE} "
            f"(remaining: {after})"
        )

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
