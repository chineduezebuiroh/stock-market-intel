# screens/style_helpers.py
import pandas as pd
from pandas.io.formats.style import Styler

def style_etf_scores(df: pd.DataFrame) -> Styler:
    """
    Shade ETF scores:
      - long_score > 0 => green-ish
      - short_score > 0 => red-ish
      - 0 / NaN => neutral
    """

    def color_long(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v) or v < 4:
                styles.append("")
            elif v >= 4:
                styles.append("background-color: rgba(0, 170, 0, 0.25); color: #006400;")
            else:
                styles.append("background-color: rgba(0, 170, 0, 0.15); color: #006400;")
        return styles

    def color_short(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v) or v == 0:
                styles.append("")
            elif v > 2:
                styles.append("background-color: rgba(220, 20, 60, 0.25); color: #8B0000;")
            else:
                styles.append("background-color: rgba(220, 20, 60, 0.15); color: #8B0000;")
        return styles

    styler = df.style

    long_cols = [
        c for c in (
            "etf_primary_long_score",
            "etf_secondary_long_score",
        ) if c in df.columns
    ]
    short_cols = [
        c for c in (
            "etf_primary_short_score",
            "etf_secondary_short_score",
        ) if c in df.columns
    ]

    if long_cols:
        styler = styler.apply(color_long, subset=long_cols)
    if short_cols:
        styler = styler.apply(color_short, subset=short_cols)

    return styler
