from pathlib import Path
import pandas as pd

def parquet_path(root: Path, timeframe: str, ticker: str) -> Path:
    return root / f"timeframe={timeframe}" / f"ticker={ticker}" / "data.parquet"

def update_fixed_window(df_new: pd.DataFrame, existing: pd.DataFrame, window_bars: int) -> pd.DataFrame:
    df = pd.concat([existing, df_new]).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if len(df) > window_bars:
        df = df.iloc[-window_bars:]
    return df
