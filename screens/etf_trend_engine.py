from __future__ import annotations

# screens/etf_trend_engine.py

from pathlib import Path
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _score_etf_row(row: pd.Series) -> tuple[float, float]:
    """
    Turn ETF indicator outputs into (etf_long_score, etf_short_score).

    This is intentionally simple and interpretable. You can tune weights later.
    Scale is roughly 0–5 in each direction.
    """
    wy = float(row.get("wyckoff_stage", 0) or 0)
    mac = float(row.get("macdv_guard", 0) or 0)
    #cloud = float(row.get("ma_trend_cloud", 0) or 0)
    sigvol = float(row.get("significant_volume", 0) or 0)

    long_score = 0.0
    short_score = 0.0

    # 1) Wyckoff bias
    if wy == 2:
        long_score += 4.0
    elif wy == -2:
        short_score += 4.0

    # 2) MACD-V (guard)
    if mac == 2:
        long_score += 2.0
    elif mac == -2:
        short_score += 2.0

    # 3) Volume confirmation
    if sigvol == 1.0:
        long_score += 1.0
        short_score += 1.0

    return long_score, short_score


def compute_etf_trend_scores(timeframe: str = "weekly") -> pd.DataFrame:
    """
    Load snapshot_etf_{timeframe}.parquet and compute ETF trend scores.

    Returns a DataFrame with index=etf_symbol and columns:
        - etf_long_score
        - etf_short_score
    """
    snap_path = DATA / f"snapshot_etf_{timeframe}.parquet"
    if not snap_path.exists():
        raise FileNotFoundError(f"Missing ETF snapshot: {snap_path}")

    snap = pd.read_parquet(snap_path)

    if "symbol" not in snap.columns:
        raise KeyError(
            f"ETF snapshot {snap_path} must contain a 'symbol' column "
            f"(have: {list(snap.columns)})"
        )

    records = []
    for _, row in snap.iterrows():
        etf_sym = str(row["symbol"])
        long_s, short_s = _score_etf_row(row)
        records.append(
            {
                "etf_symbol": etf_sym,
                "etf_long_score": float(long_s),
                "etf_short_score": float(short_s),
            }
        )

    scores = pd.DataFrame(records)
    scores = scores.drop_duplicates(subset=["etf_symbol"]).set_index("etf_symbol")
    return scores
