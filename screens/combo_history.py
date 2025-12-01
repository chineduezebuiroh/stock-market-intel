from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def combo_history_dir(data_root: Path, namespace: str, combo_name: str) -> Path:
    """
    Return the directory where history snapshots are stored for a given combo.

    Example:
        data_root / "combo_history" / "stocks" / "stocks_c_dwm_shortlist"
    """
    return data_root / "combo_history" / namespace / combo_name


def list_combo_history_files(
    data_root: Path,
    namespace: str,
    combo_name: str,
) -> list[Path]:
    """
    List all history parquet files for a given combo, sorted by timestamp
    encoded in the filename.

    Filenames look like:
        combo_stocks_stocks_c_dwm_shortlist_asof=2025-12-01T21-30-05.parquet
    """
    hist_dir = combo_history_dir(data_root, namespace, combo_name)
    if not hist_dir.exists():
        return []

    files = sorted(hist_dir.glob("combo_*_asof=*.parquet"), key=lambda p: p.name)
    return files


def load_latest_combo_history(
    data_root: Path,
    namespace: str,
    combo_name: str,
) -> Optional[pd.DataFrame]:
    """
    Load the most recent history snapshot for a combo.

    Returns:
        - A DataFrame with the same schema as the 'current' combo parquet, or
        - None if no history files exist yet.
    """
    files = list_combo_history_files(data_root, namespace, combo_name)
    if not files:
        return None

    latest = files[-1]
    return pd.read_parquet(latest)
