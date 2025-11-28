import pandas as pd


def eval_rule(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    Evaluate a simple range expression against a column.

    Expected format (examples):
      - "rsi_14 in [40, 70]"
      - "ema_8_slope in [-999, 999]"

    This function NEVER pivots or reshapes the dataframe; it just
    returns a boolean mask over the existing rows.
    """
    # Split "rsi_14 in [40, 70]"
    left, right = expr.split("in", 1)
    col = left.strip()

    # Clean "[40, 70]" → "40, 70"
    bounds = right.strip()
    bounds = bounds.strip("[]()")

    lo_str, hi_str = [x.strip() for x in bounds.split(",")]
    lo = float(lo_str)
    hi = float(hi_str)

    if col not in df.columns:
        # If the column doesn't exist, return all False
        return pd.Series(False, index=df.index)

    return (df[col] >= lo) & (df[col] <= hi)


def run_screen(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply a list of rules to `df`, adding one boolean column per rule.

    Config format:

      rules:
        - name: "rsi_mid"
          expr: "rsi_14 in [40, 60]"
        - name: "ema8_above_ema21"
          expr: "ema_8 in [ema_21, 999999]"   # (we'll keep it simple for now)

    No pivoting, no groupbys, no MultiIndex columns.
    """
    out = df.copy()

    for rule in cfg.get("rules", []):
        name = rule["name"]
        expr = rule["expr"]
        mask = eval_rule(out, expr)
        out[name] = mask

    return out
