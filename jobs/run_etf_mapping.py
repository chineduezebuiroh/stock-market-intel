from __future__ import annotations

# jobs/run_etf_mapping.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etf.mapping_engine import write_options_etf_mapping


def main():
    out = write_options_etf_mapping()
    print(f"[OK] Wrote options ETF mapping to {out}")


if __name__ == "__main__":
    main()
