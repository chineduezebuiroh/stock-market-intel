import numpy as np
import pandas as pd

def style_etf_scores(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Shade ETF scores:
      - Positive long: green-ish
      - Positive short: red-ish
      - Zero/NaN: neutral
    """

    def color_long(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v) or v == 0:
                styles.append("")
            elif v > 2:
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

    if "etf_primary_long_score" in df.columns:
        styler = styler.apply(color_long, subset=["etf_primary_long_score"])
    if "etf_primary_short_score" in df.columns:
        styler = styler.apply(color_short, subset=["etf_primary_short_score"])
    if "etf_secondary_long_score" in df.columns:
        styler = styler.apply(color_long, subset=["etf_secondary_long_score"])
    if "etf_secondary_short_score" in df.columns:
        styler = styler.apply(color_short, subset=["etf_secondary_short_score"])

    return styler
