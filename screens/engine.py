import pandas as pd


def _eval_between_expr(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    Evaluate very simple rules of the form:

        "<col> between <lo> and <hi>"

    Example:
        "rsi_14 between 40 and 60"

    Returns a boolean mask indexed like df.
    If the column doesn't exist or parsing fails, returns all False.
    """
    try:
        left, right = expr.split("between")
        col = left.strip()
        lo_str, hi_str = right.split("and")
        lo = float(lo_str)
        hi = float(hi_str)
    except Exception as e:
        print(f"[WARN] Could not parse rule expression '{expr}': {e}")
        return pd.Series(False, index=df.index)

    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found in dataframe for rule '{expr}'.")
        return pd.Series(False, index=df.index)

    return (df[col] >= lo) & (df[col] <= hi)


def run_screen(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply simple screening rules to a *row-per-symbol* dataframe.

    Expected input:
      - one row per symbol (typically last bar)
      - columns like open, high, low, close, rsi_14, etc.
      - may also contain 'symbol' column and a datetime index.

    Config format (e.g. config/screens/daily.yaml):

        rules:
          - name: "rsi_14_40_60"
            expr: "rsi_14 between 40 and 60"
          - name: "atr_14_gt_1"
            expr: "atr_14 between 1 and 9999"

    Behavior:
      - For each rule, we create a boolean mask.
      - All rules are AND-ed together to filter rows.
      - Optionally, each rule's mask can be stored in its own column.

    IMPORTANT:
      - No pivoting. We keep one row per symbol and flat column names.
    """
    if cfg is None:
        return df

    rules = cfg.get("rules", [])
    if not rules:
        return df

    out = df.copy()

    # Start with all True, then AND each rule's mask.
    combined_mask = pd.Series(True, index=out.index)

    for r in rules:
        expr = r.get("expr")
        if not expr:
            continue

        mask = _eval_between_expr(out, expr)
        combined_mask &= mask

        # If the user provided a 'name', store the mask as a column (optional)
        col_name = r.get("name")
        if col_name:
            out[col_name] = mask

    # Finally, filter down to rows that satisfy all rules
    out = out.loc[combined_mask]

    # Ensure no MultiIndex columns sneak in
    out.columns = out.columns.astype(str)

    return out
